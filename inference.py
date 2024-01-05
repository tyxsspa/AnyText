from modelscope.pipelines import pipeline
from util import save_images
pipe = pipeline('my-anytext-task', model='damo/cv_anytext_text_generation_editing', model_revision='v1.1.1')
img_save_folder = "SaveImages"
params = {
    "show_debug": True,
    "image_count": 2,
    "ddim_steps": 20,
}

# 1. text generation
mode = 'text-generation'
input_data = {
    "prompt": 'photo of caramel macchiato coffee on the table, top-down perspective, with "Any" "Text" written on it using cream',
    "seed": 66273235,
    "draw_pos": 'example_images/gen9.png'
}
results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
if rtn_code >= 0:
    save_images(results, img_save_folder)
    print(f'Done, result images are saved in: {img_save_folder}')
if rtn_warning:
    print(rtn_warning)
# 2. text editing
mode = 'text-editing'
input_data = {
    "prompt": 'A cake with colorful characters that reads "EVERYDAY"',
    "seed": 8943410,
    "draw_pos": 'example_images/edit7.png',
    "ori_image": 'example_images/ref7.jpg'
}
results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
if rtn_code >= 0:
    save_images(results, img_save_folder)
    print(f'Done, result images are saved in: {img_save_folder}')
if rtn_warning:
    print(rtn_warning)
