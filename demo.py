'''
AnyText: Multilingual Visual Text Generation And Editing
Paper: https://arxiv.org/abs/2311.03054
Code: https://github.com/tyxsspa/AnyText
Copyright (c) Alibaba, Inc. and its affiliates.
'''
import os
from modelscope.pipelines import pipeline
import cv2
import gradio as gr
import numpy as np
import re
from gradio.components import Component
from util import check_channels, resize_image, save_images

BBOX_MAX_NUM = 8
img_save_folder = 'SaveImages'
load_model = True
if load_model:
    inference = pipeline('my-anytext-task', model='damo/cv_anytext_text_generation_editing', model_revision='v1.1.0')


def count_lines(prompt):
    prompt = prompt.replace('“', '"')
    prompt = prompt.replace('”', '"')
    p = '"(.*?)"'
    strs = re.findall(p, prompt)
    if len(strs) == 0:
        strs = [' ']
    return len(strs)


class ScriptLoader:
    path_map = {
        "js": os.path.abspath(os.path.join(os.path.dirname(__file__), "javascript")),
        "py": os.path.abspath(os.path.join(os.path.dirname(__file__), "python"))
    }

    def __init__(self, script_type):
        self.script_type = script_type
        self.path = ScriptLoader.path_map[script_type]
        self.loaded_scripts = []

    @staticmethod
    def get_scripts(path: str, file_type: str) -> list[tuple[str, str]]:
        scripts = []
        dir_list = [os.path.join(path, f) for f in os.listdir(path)]
        files_list = [f for f in dir_list if os.path.isfile(f)]
        for s in files_list:
            if os.path.splitext(s)[1] == f".{file_type}":
                scripts.append((s, os.path.basename(s)))
        return scripts


class JavaScriptLoader(ScriptLoader):
    def __init__(self):
        super().__init__("js")
        self.original_template = gr.routes.templates.TemplateResponse
        self.load_js()
        gr.routes.templates.TemplateResponse = self.template_response

    def load_js(self):
        js_scripts = ScriptLoader.get_scripts(self.path,
                                              self.script_type)
        for file_path, file_name in js_scripts:
            with open(file_path, 'r', encoding="utf-8") as file:
                self.loaded_scripts.append(f"\n<!--{file_name}-->\n<script>\n{file.read()}\n</script>")

    def template_response(self, *args, **kwargs):
        response = self.original_template(*args, **kwargs)
        response.body = response.body.replace(
            '</head>'.encode('utf-8'), f"{''.join(self.loaded_scripts)}\n</head>".encode("utf-8")
        )
        response.init_headers()
        return response


def generate_rectangles(w, h, n, max_trys=200):
    img = np.zeros((h, w, 1), dtype=np.uint8)
    rectangles = []
    attempts = 0
    n_pass = 0
    low_edge = int(max(w, h)*0.3 if n <= 3 else max(w, h)*0.2)  # ~150, ~100
    while attempts < max_trys:
        rect_w = min(np.random.randint(max((w*0.5)//n, low_edge), w), int(w*0.8))
        ratio = np.random.uniform(4, 10)
        rect_h = max(low_edge, int(rect_w/ratio))
        rect_h = min(rect_h, int(h*0.8))
        # gen rotate angle
        rotation_angle = 0
        rand_value = np.random.rand()
        if rand_value < 0.7:
            pass
        elif rand_value < 0.8:
            rotation_angle = np.random.randint(0, 40)
        elif rand_value < 0.9:
            rotation_angle = np.random.randint(140, 180)
        else:
            rotation_angle = np.random.randint(85, 95)
        # rand position
        x = np.random.randint(0, w - rect_w)
        y = np.random.randint(0, h - rect_h)
        # get vertex
        rect_pts = cv2.boxPoints(((rect_w/2, rect_h/2), (rect_w, rect_h), rotation_angle))
        rect_pts = np.int32(rect_pts)
        # move
        rect_pts += (x, y)
        # check boarder
        if np.any(rect_pts < 0) or np.any(rect_pts[:, 0] >= w) or np.any(rect_pts[:, 1] >= h):
            attempts += 1
            continue
        # check overlap
        if any(check_overlap_polygon(rect_pts, rp) for rp in rectangles):
            attempts += 1
            continue
        n_pass += 1
        cv2.fillPoly(img, [rect_pts], 255)
        rectangles.append(rect_pts)
        if n_pass == n:
            break
    print("attempts:", attempts)
    if len(rectangles) != n:
        raise gr.Error(f'Failed in auto generate positions after {attempts} attempts, try again!')
    return img


def check_overlap_polygon(rect_pts1, rect_pts2):
    poly1 = cv2.convexHull(rect_pts1)
    poly2 = cv2.convexHull(rect_pts2)
    rect1 = cv2.boundingRect(poly1)
    rect2 = cv2.boundingRect(poly2)
    if rect1[0] + rect1[2] >= rect2[0] and rect2[0] + rect2[2] >= rect1[0] and rect1[1] + rect1[3] >= rect2[1] and rect2[1] + rect2[3] >= rect1[1]:
        return True
    return False


def draw_rects(width, height, rects):
    img = np.zeros((height, width, 1), dtype=np.uint8)
    for rect in rects:
        x1 = int(rect[0] * width)
        y1 = int(rect[1] * height)
        w = int(rect[2] * width)
        h = int(rect[3] * height)
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
    return img


def process(mode, prompt, pos_radio, sort_radio, revise_pos, show_debug, draw_img, rect_img, ref_img, ori_img, img_count, ddim_steps, w, h, strength, cfg_scale, seed, eta, a_prompt, n_prompt, *rect_list):
    n_lines = count_lines(prompt)
    # Text Generation
    if mode == 'gen':
        # create pos_imgs
        if pos_radio == 'Manual-draw(手绘)':
            if draw_img is not None:
                pos_imgs = 255 - draw_img['image']
                if 'mask' in draw_img:
                    pos_imgs = pos_imgs.astype(np.float32) + draw_img['mask'][..., 0:3].astype(np.float32)
                    pos_imgs = pos_imgs.clip(0, 255).astype(np.uint8)
            else:
                pos_imgs = np.zeros((w, h, 1))
        elif pos_radio == 'Manual-rect(拖框)':
            rect_check = rect_list[:BBOX_MAX_NUM]
            rect_xywh = rect_list[BBOX_MAX_NUM:]
            checked_rects = []
            for idx, c in enumerate(rect_check):
                if c:
                    _xywh = rect_xywh[4*idx:4*(idx+1)]
                    checked_rects += [_xywh]
            pos_imgs = draw_rects(w, h, checked_rects)
        elif pos_radio == 'Auto-rand(随机)':
            pos_imgs = generate_rectangles(w, h, n_lines, max_trys=500)
    # Text Editing
    elif mode == 'edit':
        revise_pos = False  # disable pos revise in edit mode
        if ref_img is None or ori_img is None:
            raise gr.Error('No reference image, please upload one for edit!')
        edit_image = ori_img.clip(1, 255)  # for mask reason
        edit_image = check_channels(edit_image)
        edit_image = resize_image(edit_image, max_length=768)
        h, w = edit_image.shape[:2]
        if isinstance(ref_img, dict) and 'mask' in ref_img and ref_img['mask'].mean() > 0:
            pos_imgs = 255 - edit_image
            edit_mask = cv2.resize(ref_img['mask'][..., 0:3], (w, h))
            pos_imgs = pos_imgs.astype(np.float32) + edit_mask.astype(np.float32)
            pos_imgs = pos_imgs.clip(0, 255).astype(np.uint8)
        else:
            if isinstance(ref_img, dict) and 'image' in ref_img:
                ref_img = ref_img['image']
            pos_imgs = 255 - ref_img  # example input ref_img is used as pos
    cv2.imwrite('pos_imgs.png', 255-pos_imgs[..., ::-1])
    params = {
        "sort_priority": sort_radio,
        "show_debug": show_debug,
        "revise_pos": revise_pos,
        "image_count": img_count,
        "ddim_steps": ddim_steps,
        "image_width": w,
        "image_height": h,
        "strength": strength,
        "cfg_scale": cfg_scale,
        "eta": eta,
        "a_prompt": a_prompt,
        "n_prompt": n_prompt
    }
    input_data = {
        "prompt": prompt,
        "seed": seed,
        "draw_pos": pos_imgs,
        "ori_image": ori_img,
    }
    results, rtn_code, rtn_warning, debug_info = inference(input_data, mode=mode, **params)
    if rtn_code >= 0:
        save_images(results, img_save_folder)
        print(f'Done, result images are saved in: {img_save_folder}')
        if rtn_warning:
            gr.Warning(rtn_warning)
    else:
        raise gr.Error(rtn_warning)
    return results, gr.Markdown(debug_info, visible=show_debug)


def create_canvas(w=512, h=512, c=3, line=5):
    image = np.full((h, w, c), 200, dtype=np.uint8)
    for i in range(h):
        if i % (w//line) == 0:
            image[i, :, :] = 150
    for j in range(w):
        if j % (w//line) == 0:
            image[:, j, :] = 150
    image[h//2-8:h//2+8, w//2-8:w//2+8, :] = [200, 0, 0]
    return image


def resize_w(w, img1, img2):
    if isinstance(img2, dict):
        img2 = img2['image']
    return [cv2.resize(img1, (w, img1.shape[0])), cv2.resize(img2, (w, img2.shape[0]))]


def resize_h(h, img1, img2):
    if isinstance(img2, dict):
        img2 = img2['image']
    return [cv2.resize(img1, (img1.shape[1], h)), cv2.resize(img2, (img2.shape[1], h))]


is_t2i = 'true'
js_loader = JavaScriptLoader()
block = gr.Blocks(css='style.css', theme=gr.themes.Soft()).queue()
with block:
    gr.HTML('<div style="text-align: center; margin: 20px auto;"> \
            <img id="banner" src="file/example_images/banner.png" alt="anytext"> <br>  \
            [<a href="https://arxiv.org/abs/2311.03054" style="color:blue; font-size:18px;">arXiv</a>] \
            [<a href="https://github.com/tyxsspa/AnyText" style="color:blue; font-size:18px;">Code</a>]\
            version: 1.1.0 </div>')
    with gr.Row(variant='compact'):
        with gr.Column():
            with gr.Accordion('🕹Instructions(说明)', open=False,):
                with gr.Tabs():
                    with gr.Tab("English"):
                        gr.Markdown('<span style="color:navy;font-size:20px">Run Examples</span>')
                        gr.Markdown('<span style="color:black;font-size:16px">AnyText has two modes: Text Generation and Text Editing, and we provides a variety of examples. Select one, click on [Run!] button to run.</span>')
                        gr.Markdown('<span style="color:gray;font-size:12px">Please note, before running examples, ensure the manual draw area is empty, otherwise may get wrong results. Additionally, different examples use \
                                     different parameters (such as resolution, seed, etc.). When generate your own, please pay attention to the parameter changes, or refresh the page to restore the default parameters.</span>')
                        gr.Markdown('<span style="color:navy;font-size:20px">Text Generation</span>')
                        gr.Markdown('<span style="color:black;font-size:16px">Enter the textual description (in Chinese or English) of the image you want to generate in [Prompt]. Each text line that needs to be generated should be \
                                     enclosed in double quotes. Then, manually draw the specified position for each text line to generate the image.</span>\
                                     <span style="color:red;font-size:16px">The drawing of text positions is crucial to the quality of the resulting image</span>, \
                                     <span style="color:black;font-size:16px">please do not draw too casually or too small. The number of positions should match the number of text lines, and the size of each position should be matched \
                                     as closely as possible to the length or width of the corresponding text line. If [Manual-draw] is inconvenient, you can try dragging rectangles [Manual-rect] or random positions [Auto-rand].</span>')
                        gr.Markdown('<span style="color:gray;font-size:12px">When generating multiple lines, each position is matched with the text line according to a certain rule. The [Sort Position] option is used to \
                                     determine whether to prioritize sorting from top to bottom or from left to right. You can open the [Show Debug] option in the parameter settings to observe the text position and glyph image \
                                     in the result. You can also select the [Revise Position] which uses the bounding box of the rendered text as the revised position. However, it is occasionally found that the creativity of the \
                                     generated text is slightly lower using this method.</span>')
                        gr.Markdown('<span style="color:navy;font-size:20px">Text Editing</span>')
                        gr.Markdown('<span style="color:black;font-size:16px">Please upload an image in [Ref] as a reference image, then adjust the brush size, and mark the area(s) to be edited. Input the textual description and \
                                     the new text to be modified in [Prompt], then generate the image.</span>')
                        gr.Markdown('<span style="color:gray;font-size:12px">The reference image can be of any resolution, but it will be internally processed with a limit that the longer side cannot exceed 768 pixels, and the \
                                     width and height will both be scaled to multiples of 64.</span>')
                    with gr.Tab("简体中文"):
                        gr.Markdown('<span style="color:navy;font-size:20px">运行示例</span>')
                        gr.Markdown('<span style="color:black;font-size:16px">AnyText有两种运行模式：文字生成和文字编辑，每种模式下提供了丰富的示例，选择一个，点击[Run!]即可。</span>')
                        gr.Markdown('<span style="color:gray;font-size:12px">请注意，运行示例前确保手绘位置区域是空的，防止影响示例结果，另外不同示例使用不同的参数（如分辨率，种子数等），如果要自行生成时，请留意参数变化，或刷新页面恢复到默认参数。</span>')
                        gr.Markdown('<span style="color:navy;font-size:20px">文字生成</span>')
                        gr.Markdown('<span style="color:black;font-size:16px">在Prompt中输入描述提示词（支持中英文），需要生成的每一行文字用双引号包裹，然后依次手绘指定每行文字的位置，生成图片。</span>\
                                     <span style="color:red;font-size:16px">文字位置的绘制对成图质量很关键</span>, \
                                     <span style="color:black;font-size:16px">请不要画的太随意或太小，位置的数量要与文字行数量一致，每个位置的尺寸要与对应的文字行的长短或宽高尽量匹配。如果手绘（Manual-draw）不方便，\
                                     可以尝试拖框矩形（Manual-rect）或随机生成（Auto-rand）。</span>')
                        gr.Markdown('<span style="color:gray;font-size:12px">多行生成时，每个位置按照一定规则排序后与文字行做对应，Sort Position选项用于确定排序时优先从上到下还是从左到右。\
                                     可以在参数设置中打开Show Debug选项，在结果图像中观察文字位置和字形图。也可以勾选Revise Position选项，这样会用渲染文字的外接矩形作为修正后的位置，不过偶尔发现这样生成的文字创造性略低。</span>')
                        gr.Markdown('<span style="color:navy;font-size:20px">文字编辑</span>')
                        gr.Markdown('<span style="color:black;font-size:16px">请上传一张待编辑的图片作为参考图(Ref)，然后调整笔触大小后，在参考图上涂抹要编辑的位置，在Prompt中输入描述提示词和要修改的文字内容，生成图片。</span>')
                        gr.Markdown('<span style="color:gray;font-size:12px">参考图可以为任意分辨率，但内部处理时会限制长边不能超过768，并且宽高都被缩放为64的整数倍。</span>')
            with gr.Accordion('🛠Parameters(参数)', open=False):
                with gr.Row(variant='compact'):
                    img_count = gr.Slider(label="Image Count(图片数)", minimum=1, maximum=12, value=4, step=1)
                    ddim_steps = gr.Slider(label="Steps(步数)", minimum=1, maximum=100, value=20, step=1)
                with gr.Row(variant='compact'):
                    image_width = gr.Slider(label="Image Width(宽度)", minimum=256, maximum=768, value=512, step=64)
                    image_height = gr.Slider(label="Image Height(高度)", minimum=256, maximum=768, value=512, step=64)
                with gr.Row(variant='compact'):
                    strength = gr.Slider(label="Strength(控制力度)", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                    cfg_scale = gr.Slider(label="CFG-Scale(CFG强度)", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                with gr.Row(variant='compact'):
                    seed = gr.Slider(label="Seed(种子数)", minimum=-1, maximum=99999999, step=1, randomize=False, value=-1)
                    eta = gr.Number(label="eta (DDIM)", value=0.0)
                with gr.Row(variant='compact'):
                    show_debug = gr.Checkbox(label='Show Debug(调试信息)', value=False)
                    gr.Markdown('<span style="color:silver;font-size:12px">whether show glyph image and debug information in the result(是否在结果中显示glyph图以及调试信息)</span>')
                a_prompt = gr.Textbox(label="Added Prompt(附加提示词)", value='best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks')
                n_prompt = gr.Textbox(label="Negative Prompt(负向提示词)", value='low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture')
            prompt = gr.Textbox(label="Prompt(提示词)")
            with gr.Tabs() as tab_modes:
                with gr.Tab("🖼Text Generation(文字生成)", elem_id='MD-tab-t2i') as mode_gen:
                    pos_radio = gr.Radio(["Manual-draw(手绘)", "Manual-rect(拖框)", "Auto-rand(随机)"], value='Manual-draw(手绘)', label="Pos-Method(位置方式)", info="choose a method to specify text positions(选择方法用于指定文字位置).")
                    with gr.Row():
                        sort_radio = gr.Radio(["↕", "↔"], value='↕', label="Sort Position(位置排序)", info="position sorting priority(位置排序时的优先级)")
                        revise_pos = gr.Checkbox(label='Revise Position(修正位置)', value=False)
                        # gr.Markdown('<span style="color:silver;font-size:12px">try to revise according to text\'s bounding rectangle(尝试通过渲染后的文字行的外接矩形框修正位置)</span>')
                    with gr.Row(variant='compact'):
                        rect_cb_list: list[Component] = []
                        rect_xywh_list: list[Component] = []
                        for i in range(BBOX_MAX_NUM):
                            e = gr.Checkbox(label=f'{i}', value=False, visible=False, min_width='10')
                            x = gr.Slider(label='x', value=0.4, minimum=0.0, maximum=1.0, step=0.0001, elem_id=f'MD-t2i-{i}-x', visible=False)
                            y = gr.Slider(label='y', value=0.4, minimum=0.0, maximum=1.0, step=0.0001, elem_id=f'MD-t2i-{i}-y',  visible=False)
                            w = gr.Slider(label='w', value=0.2, minimum=0.0, maximum=1.0, step=0.0001, elem_id=f'MD-t2i-{i}-w',  visible=False)
                            h = gr.Slider(label='h', value=0.2, minimum=0.0, maximum=1.0, step=0.0001, elem_id=f'MD-t2i-{i}-h',  visible=False)
                            x.change(fn=None, inputs=x, outputs=x, _js=f'v => onBoxChange({is_t2i}, {i}, "x", v)', show_progress=False)
                            y.change(fn=None, inputs=y, outputs=y, _js=f'v => onBoxChange({is_t2i}, {i}, "y", v)', show_progress=False)
                            w.change(fn=None, inputs=w, outputs=w, _js=f'v => onBoxChange({is_t2i}, {i}, "w", v)', show_progress=False)
                            h.change(fn=None, inputs=h, outputs=h, _js=f'v => onBoxChange({is_t2i}, {i}, "h", v)', show_progress=False)

                            e.change(fn=None, inputs=e, outputs=e, _js=f'e => onBoxEnableClick({is_t2i}, {i}, e)')
                            rect_cb_list.extend([e])
                            rect_xywh_list.extend([x, y, w, h])

                    rect_img = gr.Image(value=create_canvas(), label="Rext Position(方框位置)", elem_id="MD-bbox-rect-t2i", show_label=False, visible=False)
                    draw_img = gr.Image(value=create_canvas(), label="Draw Position(绘制位置)", visible=True, tool='sketch', show_label=False, brush_radius=60)

                    def re_draw():
                        return [gr.Image(value=create_canvas(), tool='sketch'), gr.Slider(value=512), gr.Slider(value=512)]
                    draw_img.clear(re_draw, None, [draw_img, image_width, image_height])
                    image_width.release(resize_w, [image_width, rect_img, draw_img], [rect_img, draw_img])
                    image_height.release(resize_h, [image_height, rect_img, draw_img], [rect_img, draw_img])

                    def change_options(selected_option):
                        return [gr.Checkbox(visible=selected_option == 'Manual-rect(拖框)')] * BBOX_MAX_NUM + \
                                [gr.Image(visible=selected_option == 'Manual-rect(拖框)'),
                                 gr.Image(visible=selected_option == 'Manual-draw(手绘)'),
                                 gr.Radio(visible=selected_option != 'Auto-rand(随机)'),
                                 gr.Checkbox(value=selected_option == 'Auto-rand(随机)')]
                    pos_radio.change(change_options, pos_radio, rect_cb_list + [rect_img, draw_img, sort_radio, revise_pos], show_progress=False)
                    with gr.Row():
                        gr.Markdown("")
                        run_gen = gr.Button(value="Run(运行)!", scale=0.3, elem_classes='run')
                        gr.Markdown("")

                    def exp_gen_click():
                        return [gr.Slider(value=512), gr.Slider(value=512)]  # all examples are 512x512, refresh draw_img
                    exp_gen = gr.Examples(
                        [
                            ['一只浣熊站在黑板前，上面写着"深度学习"', "example_images/gen1.png", "Manual-draw(手绘)", "↕", False, 4, 81808278],
                            ['一个儿童蜡笔画，森林里有一个可爱的蘑菇形状的房子，标题是"森林小屋"', "example_images/gen16.png", "Manual-draw(手绘)", "↕", False, 4, 40173333],
                            ['一个精美设计的logo，画的是一个黑白风格的厨师，带着厨师帽，logo下方写着“深夜食堂”', "example_images/gen14.png", "Manual-draw(手绘)", "↕", False, 4, 6970544],
                            ['photo of caramel macchiato coffee on the table, top-down perspective, with "Any" "Text" written on it using cream', "example_images/gen9.png", "Manual-draw(手绘)", "↕", False, 4, 66273235],
                            ['一张户外雪地靴的电商广告，上面写着 “双12大促！”，“立减50”，“加绒加厚”，“穿脱方便”，“温暖24小时送达”， “包邮”，高级设计感，精美构图', "example_images/gen15.png", "Manual-draw(手绘)", "↕", False, 4, 66980376],
                            ['Sign on the clean building that reads "科学" and "과학"  and "ステップ" and "SCIENCE"', "example_images/gen6.png", "Manual-draw(手绘)", "↕", True, 4, 13246309],
                            ['一个精致的马克杯，上面雕刻着一首中国古诗，内容是 "花落知多少" "夜来风雨声" "处处闻啼鸟" "春眠不觉晓"', "example_images/gen3.png", "Manual-draw(手绘)", "↔", False, 4, 60358279],
                            ['A delicate square cake, cream and fruit, with "CHEERS" "to the" and "GRADUATE" written in chocolate', "example_images/gen8.png", "Manual-draw(手绘)", "↕", False, 4, 93424638],
                            ['一件精美的毛衣，上面有针织的文字："通义丹青"', "example_images/gen4.png", "Manual-draw(手绘)", "↕", False, 4, 48769450],
                            ['一个双肩包的特写照，上面用针织文字写着”为了无法“ ”计算的价值“', "example_images/gen12.png", "Manual-draw(手绘)", "↕", False, 4, 35552323],
                            ['A nice drawing in pencil of Michael Jackson,  with the words "Micheal" and "Jackson" written on it', "example_images/gen7.png", "Manual-draw(手绘)", "↕", False, 4, 83866922],
                            ['一个漂亮的蜡笔画，有行星，宇航员，还有宇宙飞船，上面写的是"去火星旅行", "王小明", "11月1日"', "example_images/gen5.png", "Manual-draw(手绘)", "↕", False, 4, 42328250],
                            ['一个装饰华丽的蛋糕，上面用奶油写着“阿里云”和"APSARA"', "example_images/gen13.png", "Manual-draw(手绘)", "↕", False, 4, 62357019],
                            ['一张关于墙上的彩色涂鸦艺术的摄影作品，上面写着“人工智能" 和 "神经网络"', "example_images/gen10.png", "Manual-draw(手绘)", "↕", False, 4, 64722007],
                            ['一枚中国古代铜钱,  上面的文字是 "康"  "寶" "通" "熙"', "example_images/gen2.png", "Manual-draw(手绘)", "↕", False, 4, 24375031],
                            ['a well crafted ice sculpture that made with "Happy" and "Holidays". Dslr photo, perfect illumination', "example_images/gen11.png", "Manual-draw(手绘)", "↕", True, 4, 64901362],
                        ],
                        [prompt, draw_img, pos_radio, sort_radio, revise_pos, img_count, seed],
                        examples_per_page=5,
                    )
                    exp_gen.dataset.click(exp_gen_click, None, [image_width, image_height])

                with gr.Tab("🎨Text Editing(文字编辑)") as mode_edit:
                    with gr.Row(variant='compact'):
                        ref_img = gr.Image(label='Ref(参考图)', source='upload')
                        ori_img = gr.Image(label='Ori(原图)')

                    def upload_ref(x):
                        return [gr.Image(type="numpy", brush_radius=60, tool='sketch'),
                                gr.Image(value=x)]

                    def clear_ref(x):
                        return gr.Image(source='upload', tool=None)
                    ref_img.upload(upload_ref, ref_img, [ref_img, ori_img])
                    ref_img.clear(clear_ref, ref_img, ref_img)
                    with gr.Row():
                        gr.Markdown("")
                        run_edit = gr.Button(value="Run(运行)!", scale=0.3, elem_classes='run')
                        gr.Markdown("")
                    gr.Examples(
                        [
                            ['精美的书法作品，上面写着“志” “存” “高” ”远“', "example_images/ref10.jpg", "example_images/edit10.png", 4, 98053044],
                            ['一个表情包，小猪说 "下班"', "example_images/ref2.jpg", "example_images/edit2.png", 2, 43304008],
                            ['Characters written in chalk on the blackboard that says "DADDY"', "example_images/ref8.jpg", "example_images/edit8.png", 4, 73556391],
                            ['一个中国古代铜钱，上面写着"乾" "隆"', "example_images/ref12.png", "example_images/edit12.png", 4, 89159482],
                            ['黑板上写着"Here"', "example_images/ref11.jpg", "example_images/edit11.png", 2, 15353513],
                            ['A letter picture that says "THER"', "example_images/ref6.jpg", "example_images/edit6.png", 4, 72321415],
                            ['一堆水果, 中间写着“UIT”', "example_images/ref13.jpg", "example_images/edit13.png", 4, 54263567],
                            ['一个漫画，上面写着" "', "example_images/ref14.png", "example_images/edit14.png", 4, 94081527],
                            ['一个黄色标志牌，上边写着"不要" 和 "大意"', "example_images/ref3.jpg", "example_images/edit3.png", 2, 64010349],
                            ['A cake with colorful characters that reads "EVERYDAY"', "example_images/ref7.jpg", "example_images/edit7.png", 4, 8943410],
                            ['一个青铜鼎，上面写着"  "和"  "', "example_images/ref4.jpg", "example_images/edit4.png", 4, 71139289],
                            ['一个建筑物前面的字母标牌， 上面写着 " "', "example_images/ref5.jpg", "example_images/edit5.png", 4, 50416289],
                        ],
                        [prompt, ori_img, ref_img, img_count, seed],
                        examples_per_page=5,
                    )
        with gr.Column():
            result_gallery = gr.Gallery(label='Result(结果)', show_label=True, preview=True, columns=2, allow_preview=True, height=600)
            result_info = gr.Markdown('', visible=False)
    ips = [prompt, pos_radio, sort_radio, revise_pos, show_debug, draw_img, rect_img, ref_img, ori_img, img_count, ddim_steps, image_width, image_height, strength, cfg_scale, seed, eta, a_prompt, n_prompt, *(rect_cb_list+rect_xywh_list)]
    run_gen.click(fn=process, inputs=[gr.State('gen')] + ips, outputs=[result_gallery, result_info])
    run_edit.click(fn=process, inputs=[gr.State('edit')] + ips, outputs=[result_gallery, result_info])

block.launch(
    server_name='0.0.0.0' if os.getenv('GRADIO_LISTEN', '') != '' else "127.0.0.1",
    share=False,
    root_path=f"/{os.getenv('GRADIO_PROXY_PATH')}" if os.getenv('GRADIO_PROXY_PATH') else ""
)
# block.launch(server_name='0.0.0.0')
