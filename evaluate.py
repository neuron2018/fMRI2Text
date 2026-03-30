import torch
import vec2text
import pandas as pd
import webdataset as wds
import generation_eval_utils
from torch.utils.data import DataLoader
from model import X_high
from tqdm import tqdm


def get_valid_iter(subj, batch_size):
    val_url = f"nsd_data/webdataset_avg_new/test/test_subj{subj:02}_" + "{0..1}.tar"

    valid_data = (
        wds.WebDataset(val_url)
        .decode("torch")
        .rename(images="jpg;png", voxels="nsdgeneral.npy",
                trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")
        .to_tuple(*["coco", "voxels", "images"])
        .batched(batch_size)
    )

    valid_iter = DataLoader(valid_data, batch_size=None, shuffle=False)

    return valid_iter


model = X_high(hidden_dim=1536, num_latents=4, depth=4)
checkpoint = torch.load(f'./model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval().cuda()

corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")

captions = [list(pd.read_csv(f'./captions.csv')[f'caption{i+1}']) for i in range(5)]
valid_iter = get_valid_iter(subj=1, batch_size=100)

with torch.no_grad():

    indices, references = [], []
    predictions, images = [], []

    for coco, voxel, image in tqdm(valid_iter):
        indices.extend(list(map(int, coco.squeeze().tolist())))

        images.append(image)

        voxel = torch.mean(voxel, dim=1).float().cuda()
        emb_voxel, _ = model(voxel, modal=f"fmri1")
        emb_voxel = emb_voxel.squeeze()

        prediction = vec2text.invert_embeddings(
            embeddings=emb_voxel.cuda(),
            corrector=corrector,
            num_steps=20,
        )

        predictions.extend(prediction)

    for i in indices:
        references.append([captions[j][i] for j in range(5)])

    images = torch.vstack(images)

headers = ['index', 'preds']
result = list(zip(indices, predictions))
output = pd.DataFrame(result, columns=headers)
output.to_csv(f'./result.csv', index=False)

metrics = generation_eval_utils.get_all_metrics(references, predictions, images)
