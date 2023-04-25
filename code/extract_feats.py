import torch
import thingsvision.vision as vision

from thingsvision.model_class import Model


def extract_features(image_folder, out_folder, model_name = 'clip-ViT',backend = 'pt',batch_size = 64):


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model(model_name, pretrained=True, model_path=None, device=device, backend=backend)
    module_name = model.show()
    dl = vision.load_dl(
        root= f'../data/{image_path}',
        out_path=f'../data/{model_name}/{module_name}/features',
        batch_size=batch_size,
        transforms=model.get_transformations(),
        backend=backend,
        )
    features, targets = model.extract_features(
                data_loader=dl,
                module_name=module_name,
                flatten_acts=False,
                clip=True,
                )
    features = vision.center_features(features)

    vision.save_features(features, f'./{model_name}/{module_name}/features', 'npy')
    vision.save_targets(targets, f'./{model_name}/{module_name}/targets', 'npy')