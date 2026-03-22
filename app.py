"""
MiniGPT-v2 PyInstaller entry point.

Sets up paths correctly whether running as a normal Python script or as a frozen exe.
"""
import os
import sys
import multiprocessing
import argparse
import random

# Determine base directory early, before any other imports
def get_base_dir():
    """Get the base directory - exe folder for frozen, repo root for normal."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
os.chdir(BASE_DIR)

# For PyInstaller frozen exe, ensure packages can be found
if getattr(sys, 'frozen', False):
    meipass = sys._MEIPASS
    if meipass not in sys.path:
        sys.path.insert(0, meipass)

# Fix Windows temp dir for gradio image saving
_tmp_dir = os.path.join(BASE_DIR, "tmp")
os.makedirs(_tmp_dir, exist_ok=True)
os.environ.setdefault("TMPDIR", _tmp_dir)

# Clean stale temp files from previous runs
for _f in os.listdir(_tmp_dir):
    _fp = os.path.join(_tmp_dir, _f)
    try:
        if os.path.isfile(_fp) and _f.endswith(".jpg"):
            os.remove(_fp)
    except OSError:
        pass

# Now import heavy dependencies and minigpt4 (triggers registry)
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# These wildcard imports register models/processors/tasks in the registry
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def _print_startup_info():
    """Print diagnostic info at startup."""
    print("=" * 60)
    print("MiniGPT-v2 Desktop Application")
    print("=" * 60)
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM: {vram_gb:.1f} GB")
    print(f"  Frozen exe: {getattr(sys, 'frozen', False)}")
    print("=" * 60)


def _validate_paths(cfg_path, base_dir):
    """Validate that required files exist, return list of errors."""
    errors = []
    if not os.path.isfile(cfg_path):
        errors.append(f"Config file not found: {cfg_path}")
    return errors


def main():
    base_dir = BASE_DIR

    # Parse args - set defaults for exe usage
    parser = argparse.ArgumentParser(description="MiniGPT-v2 Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigptv2_eval.yaml",
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--options", nargs="+",
                        help="override settings in config, key=value format.")
    args = parser.parse_args()

    # Resolve cfg-path relative to base_dir
    if not os.path.isabs(args.cfg_path):
        args.cfg_path = os.path.join(base_dir, args.cfg_path)

    # Also check _internal for PyInstaller bundled configs
    if not os.path.isfile(args.cfg_path) and getattr(sys, 'frozen', False):
        internal_path = os.path.join(base_dir, "_internal", "eval_configs",
                                      os.path.basename(args.cfg_path))
        if os.path.isfile(internal_path):
            args.cfg_path = internal_path

    _print_startup_info()

    # Validate CUDA
    if not torch.cuda.is_available():
        print("\nERROR: No CUDA GPU detected!")
        print("MiniGPT-v2 requires an NVIDIA GPU with CUDA support.")
        print("Please ensure:")
        print("  1. You have an NVIDIA GPU")
        print("  2. NVIDIA drivers are installed")
        print("  3. CUDA toolkit is available")
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Validate paths
    path_errors = _validate_paths(args.cfg_path, base_dir)
    if path_errors:
        print("\nERROR: Missing required files:")
        for err in path_errors:
            print(f"  - {err}")
        input("\nPress Enter to exit...")
        sys.exit(1)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cudnn.benchmark = False
    cudnn.deterministic = True

    print('\nInitializing Chat...')
    try:
        cfg = Config(args)
    except Exception as e:
        print(f"\nERROR: Failed to load config: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)

    device = 'cuda:{}'.format(args.gpu_id)

    # Validate model files exist before attempting to load
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id

    llama_path = getattr(model_config, 'llama_model', None)
    if llama_path and not os.path.isabs(llama_path):
        llama_path = os.path.join(base_dir, llama_path)
    if llama_path and not os.path.isdir(llama_path):
        print(f"\nERROR: LLM model not found at: {llama_path}")
        print("Please download Llama-2-7b-chat-hf and place it in the models/ directory.")
        input("\nPress Enter to exit...")
        sys.exit(1)

    ckpt_path = getattr(model_config, 'ckpt', None)
    if ckpt_path and not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(base_dir, ckpt_path)
    if ckpt_path and not os.path.isfile(ckpt_path):
        print(f"\nERROR: Model checkpoint not found at: {ckpt_path}")
        print("Please download the MiniGPT-v2 checkpoint and place it in the models/ directory.")
        input("\nPress Enter to exit...")
        sys.exit(1)

    try:
        model_cls = registry.get_model_class(model_config.arch)
        print(f"  Loading model: {model_config.arch}")
        model = model_cls.from_config(model_config).to(device)
        model = model.eval()
    except torch.cuda.OutOfMemoryError:
        print(f"\nERROR: Not enough GPU memory to load the model!")
        print(f"  Available VRAM: {torch.cuda.get_device_properties(args.gpu_id).total_mem / (1024**3):.1f} GB")
        print("  Try enabling low_resource mode in the config (uses 8-bit quantization).")
        input("\nPress Enter to exit...")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    chat = Chat(model, vis_processor, device=device)

    print("\nModel loaded successfully! Starting Gradio UI...")
    _run_gradio(chat, base_dir)


def _run_gradio(chat, base_dir):
    """Launch the Gradio demo UI."""
    import re
    import html
    import random
    from collections import defaultdict

    import cv2
    import numpy as np
    from PIL import Image
    import torch
    import torchvision.transforms as T
    import gradio as gr

    from minigpt4.conversation.conversation import Conversation, SeparatorStyle

    bounding_box_size = 100

    CONV_VISION = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )

    def extract_substrings(string):
        index = string.rfind('}')
        if index != -1:
            string = string[:index + 1]
        pattern = r'<p>(.*?)\}(?!<)'
        matches = re.findall(pattern, string)
        return [match for match in matches]

    def is_overlapping(rect1, rect2):
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

    def computeIoU(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        intersection_x1 = max(x1, x3)
        intersection_y1 = max(y1, y3)
        intersection_x2 = min(x2, x4)
        intersection_y2 = min(y2, y4)
        intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
        bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
        union_area = bbox1_area + bbox2_area - intersection_area
        iou = intersection_area / union_area
        return iou

    _tmp_files = []
    _MAX_TMP_FILES = 50

    def save_tmp_img(visual_img):
        file_name = "".join([str(random.randint(0, 9)) for _ in range(5)]) + ".jpg"
        tmp_dir = os.path.join(base_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        file_path = os.path.join(tmp_dir, file_name)
        visual_img.save(file_path)
        _tmp_files.append(file_path)
        # Clean up old temp files to prevent unbounded growth
        while len(_tmp_files) > _MAX_TMP_FILES:
            old_file = _tmp_files.pop(0)
            try:
                if os.path.exists(old_file):
                    os.remove(old_file)
            except OSError:
                pass
        return file_path

    def mask2bbox(mask):
        if mask is None:
            return ''
        mask = mask.resize([100, 100], resample=Image.NEAREST)
        mask = np.array(mask)[:, :, 0]
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.sum():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox = '{{<{}><{}><{}><{}>}}'.format(cmin, rmin, cmax, rmax)
        else:
            bbox = ''
        return bbox

    def escape_markdown(text):
        md_chars = ['<', '>']
        for char in md_chars:
            text = text.replace(char, '\\' + char)
        return text

    def reverse_escape(text):
        md_chars = ['\\<', '\\>']
        for char in md_chars:
            text = text.replace(char, char[1:])
        return text

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (210, 210, 0),
        (255, 0, 255), (0, 255, 255), (114, 128, 250), (0, 165, 255),
        (0, 128, 0), (144, 238, 144), (238, 238, 175), (255, 191, 0),
        (0, 128, 0), (226, 43, 138), (255, 0, 255), (0, 215, 255),
    ]

    color_map = {
        f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}"
        for color_id, color in enumerate(colors)
    }

    def visualize_all_bbox_together(image, generation):
        if image is None:
            return None, ''
        generation = html.unescape(generation)
        image_width, image_height = image.size
        image = image.resize([500, int(500 / image_width * image_height)])
        image_width, image_height = image.size

        string_list = extract_substrings(generation)
        if string_list:
            mode = 'all'
            entities = defaultdict(list)
            i = 0
            j = 0
            for string in string_list:
                try:
                    obj, string = string.split('</p>')
                except ValueError:
                    print('wrong string: ', string)
                    continue
                bbox_list = string.split('<delim>')
                flag = False
                for bbox_string in bbox_list:
                    integers = re.findall(r'-?\d+', bbox_string)
                    if len(integers) == 4:
                        x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
                        left = x0 / bounding_box_size * image_width
                        bottom = y0 / bounding_box_size * image_height
                        right = x1 / bounding_box_size * image_width
                        top = y1 / bounding_box_size * image_height
                        entities[obj].append([left, bottom, right, top])
                        j += 1
                        flag = True
                if flag:
                    i += 1
        else:
            integers = re.findall(r'-?\d+', generation)
            if len(integers) == 4:
                mode = 'single'
                entities = list()
                x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
                left = x0 / bounding_box_size * image_width
                bottom = y0 / bounding_box_size * image_height
                right = x1 / bounding_box_size * image_width
                top = y1 / bounding_box_size * image_height
                entities.append([left, bottom, right, top])
            else:
                return None, ''

        if len(entities) == 0:
            return None, ''

        if isinstance(image, Image.Image):
            image_h = image.height
            image_w = image.width
            image = np.array(image)
        elif isinstance(image, str):
            if os.path.exists(image):
                pil_img = Image.open(image).convert("RGB")
                image = np.array(pil_img)[:, :, [2, 1, 0]]
                image_h = pil_img.height
                image_w = pil_img.width
            else:
                raise ValueError(f"invalid image path, {image}")
        elif isinstance(image, torch.Tensor):
            image_tensor = image.cpu()
            reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
            reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
            image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
            pil_img = T.ToPILImage()(image_tensor)
            image_h = pil_img.height
            image_w = pil_img.width
            image = np.array(pil_img)[:, :, [2, 1, 0]]
        else:
            raise ValueError(f"invalid image format, {type(image)} for {image}")

        indices = list(range(len(entities)))
        new_image = image.copy()
        previous_bboxes = []
        text_size = 0.5
        text_line = 1
        box_line = 2
        (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
        base_height = int(text_height * 0.675)
        text_offset_original = text_height - base_height
        text_spaces = 2
        used_colors = colors

        color_id = -1
        for entity_idx, entity_name in enumerate(entities):
            if mode == 'single' or mode == 'identify':
                bboxes = entity_name
                bboxes = [bboxes]
            else:
                bboxes = entities[entity_name]
            color_id += 1
            for bbox_id, (x1_norm, y1_norm, x2_norm, y2_norm) in enumerate(bboxes):
                skip_flag = False
                orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm), int(y1_norm), int(x2_norm), int(y2_norm)
                color = used_colors[entity_idx % len(used_colors)]
                new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

                if mode == 'all':
                    l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1
                    x1 = orig_x1 - l_o
                    y1 = orig_y1 - l_o
                    if y1 < text_height + text_offset_original + 2 * text_spaces:
                        y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                        x1 = orig_x1 + r_o
                    (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
                    text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

                    for prev_bbox in previous_bboxes:
                        if computeIoU((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']) > 0.95 and prev_bbox['phrase'] == entity_name:
                            skip_flag = True
                            break
                        while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']):
                            text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                            text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                            y1 += (text_height + text_offset_original + 2 * text_spaces)
                            if text_bg_y2 >= image_h:
                                text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                                text_bg_y2 = image_h
                                y1 = image_h
                                break
                    if not skip_flag:
                        alpha = 0.5
                        for i in range(text_bg_y1, text_bg_y2):
                            for j in range(text_bg_x1, text_bg_x2):
                                if i < image_h and j < image_w:
                                    if j < text_bg_x1 + 1.35 * c_width:
                                        bg_color = color
                                    else:
                                        bg_color = [255, 255, 255]
                                    new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(np.uint8)
                        cv2.putText(new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces),
                                    cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA)
                        previous_bboxes.append({'bbox': (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), 'phrase': entity_name})

        if mode == 'all':
            def color_iterator(colors):
                while True:
                    for color in colors:
                        yield color
            color_gen = color_iterator(colors)

            def colored_phrases(match):
                phrase = match.group(1)
                color = next(color_gen)
                return f'<span style="color:rgb{color}">{phrase}</span>'

            generation = re.sub(r'{<\d+><\d+><\d+><\d+>}|<delim>', '', generation)
            generation_colored = re.sub(r'<p>(.*?)</p>', colored_phrases, generation)
        else:
            generation_colored = ''

        pil_image = Image.fromarray(new_image)
        return pil_image, generation_colored

    def gradio_reset(chat_state, img_list):
        if chat_state is not None:
            chat_state.messages = []
        if img_list is not None:
            img_list = []
        return None, gr.update(value=None, interactive=True), gr.update(placeholder='Upload your image and chat', interactive=True), chat_state, img_list

    def image_upload_trigger(upload_flag, replace_flag, img_list):
        upload_flag = 1
        if img_list:
            replace_flag = 1
        return upload_flag, replace_flag

    def example_trigger(text_input, image, upload_flag, replace_flag, img_list):
        upload_flag = 1
        if img_list or replace_flag == 1:
            replace_flag = 1
        return upload_flag, replace_flag

    def gradio_ask(user_message, chatbot, chat_state, gr_img, img_list, upload_flag, replace_flag):
        if len(user_message) == 0:
            text_box_show = 'Input should not be empty!'
        else:
            text_box_show = ''

        if isinstance(gr_img, dict):
            gr_img, mask = gr_img['image'], gr_img['mask']
        else:
            mask = None

        if '[identify]' in user_message:
            integers = re.findall(r'-?\d+', user_message)
            if len(integers) != 4:
                bbox = mask2bbox(mask)
                user_message = user_message + bbox

        if chat_state is None:
            chat_state = CONV_VISION.copy()

        if upload_flag:
            if replace_flag:
                chat_state = CONV_VISION.copy()
                replace_flag = 0
                chatbot = []
            img_list = []
            llm_message = chat.upload_img(gr_img, chat_state, img_list)
            upload_flag = 0

        chat.ask(user_message, chat_state)
        chatbot = chatbot + [[user_message, None]]

        if '[identify]' in user_message:
            visual_img, _ = visualize_all_bbox_together(gr_img, user_message)
            if visual_img is not None:
                file_path = save_tmp_img(visual_img)
                chatbot = chatbot + [[(file_path,), None]]

        return text_box_show, chatbot, chat_state, img_list, upload_flag, replace_flag

    def gradio_stream_answer(chatbot, chat_state, img_list, temperature):
        if len(img_list) > 0:
            if not isinstance(img_list[0], torch.Tensor):
                chat.encode_img(img_list)

        streamer = chat.stream_answer(conv=chat_state,
                                      img_list=img_list,
                                      temperature=temperature,
                                      max_new_tokens=500,
                                      max_length=2000)
        output = ''
        for new_output in streamer:
            escapped = escape_markdown(new_output)
            output += escapped
            chatbot[-1][1] = output
            yield chatbot, chat_state
        chat_state.messages[-1][1] = '</s>'
        return chatbot, chat_state

    def gradio_visualize(chatbot, gr_img):
        if isinstance(gr_img, dict):
            gr_img, mask = gr_img['image'], gr_img['mask']
        unescaped = reverse_escape(chatbot[-1][1])
        visual_img, generation_color = visualize_all_bbox_together(gr_img, unescaped)
        if visual_img is not None:
            if len(generation_color):
                chatbot[-1][1] = generation_color
            file_path = save_tmp_img(visual_img)
            chatbot = chatbot + [[None, (file_path,)]]
        return chatbot

    def gradio_taskselect(idx):
        prompt_list = [
            '',
            '[grounding] describe this image in detail',
            '[refer] ',
            '[detection] ',
            '[identify] what is this ',
            '[vqa] '
        ]
        instruct_list = [
            '**Hint:** Type in whatever you want',
            '**Hint:** Send the command to generate a grounded image description',
            '**Hint:** Type in a phrase about an object in the image and send the command',
            '**Hint:** Type in a caption or phrase, and see object locations in the image',
            '**Hint:** Draw a bounding box on the uploaded image then send the command.',
            '**Hint:** Send a question to get a short answer',
        ]
        return prompt_list[idx], instruct_list[idx]

    # Build Gradio UI
    title = """<h1 align="center">MiniGPT-v2 Demo</h1>"""
    article = """<p><a href='https://minigpt-v2.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p>"""
    introduction = '''
For Abilities Involving Visual Grounding:
1. Grounding: CLICK **Send** to generate a grounded image description.
2. Refer: Input a referring object and CLICK **Send**.
3. Detection: Write a caption or phrase, and CLICK **Send**.
4. Identify: Draw the bounding box on the uploaded image window and CLICK **Send**.
5. VQA: Input a visual question and CLICK **Send**.
6. No Tag: Input whatever you want and CLICK **Send** without any tagging
'''

    text_input = gr.Textbox(placeholder='Upload your image and chat', interactive=True, show_label=False, container=False, scale=8)

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(article)

        with gr.Row():
            with gr.Column(scale=0.5):
                image = gr.Image(type="pil", tool='sketch', brush_radius=20)
                temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.6, step=0.1, interactive=True, label="Temperature")
                clear = gr.Button("Restart")
                gr.Markdown(introduction)

            with gr.Column():
                chat_state = gr.State(value=None)
                img_list = gr.State(value=[])
                chatbot = gr.Chatbot(label='MiniGPT-v2')

                dataset = gr.Dataset(
                    components=[gr.Textbox(visible=False)],
                    samples=[['No Tag'], ['Grounding'], ['Refer'], ['Detection'], ['Identify'], ['VQA']],
                    type="index",
                    label='Task Shortcuts',
                )
                task_inst = gr.Markdown('**Hint:** Upload your image and chat')
                with gr.Row():
                    text_input.render()
                    send = gr.Button("Send", variant='primary', size='sm', scale=1)

        upload_flag = gr.State(value=0)
        replace_flag = gr.State(value=0)
        image.upload(image_upload_trigger, [upload_flag, replace_flag, img_list], [upload_flag, replace_flag])

        # Check for example images
        examples_dir = os.path.join(base_dir, "examples_v2")
        if os.path.isdir(examples_dir):
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=[
                        [os.path.join(examples_dir, "office.jpg"), "[grounding] describe this image in detail", upload_flag, replace_flag, img_list],
                        [os.path.join(examples_dir, "sofa.jpg"), "[detection] sofas", upload_flag, replace_flag, img_list],
                    ], inputs=[image, text_input, upload_flag, replace_flag, img_list], fn=example_trigger, outputs=[upload_flag, replace_flag])
                with gr.Column():
                    gr.Examples(examples=[
                        [os.path.join(examples_dir, "glip_test.jpg"), "[vqa] where should I hide in this room when playing hide and seek", upload_flag, replace_flag, img_list],
                        [os.path.join(examples_dir, "float.png"), "Please write a poem about the image", upload_flag, replace_flag, img_list],
                    ], inputs=[image, text_input, upload_flag, replace_flag, img_list], fn=example_trigger, outputs=[upload_flag, replace_flag])

        dataset.click(gradio_taskselect, inputs=[dataset], outputs=[text_input, task_inst], show_progress="hidden", postprocess=False, queue=False)

        text_input.submit(
            gradio_ask, [text_input, chatbot, chat_state, image, img_list, upload_flag, replace_flag],
            [text_input, chatbot, chat_state, img_list, upload_flag, replace_flag], queue=False
        ).success(
            gradio_stream_answer, [chatbot, chat_state, img_list, temperature], [chatbot, chat_state]
        ).success(
            gradio_visualize, [chatbot, image], [chatbot], queue=False
        )

        send.click(
            gradio_ask, [text_input, chatbot, chat_state, image, img_list, upload_flag, replace_flag],
            [text_input, chatbot, chat_state, img_list, upload_flag, replace_flag], queue=False
        ).success(
            gradio_stream_answer, [chatbot, chat_state, img_list, temperature], [chatbot, chat_state]
        ).success(
            gradio_visualize, [chatbot, image], [chatbot], queue=False
        )

        clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, chat_state, img_list], queue=False)

    demo.launch(share=False, enable_queue=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for PyInstaller on Windows
    main()
