import os
import skillsnetwork
import asyncio

from variables_set import data_dir 
from variables_set import dataset_url 

from def_frames import check_skillnetwork_extraction
from def_frames import download_tar_dataset
from def_frames import download_model


async def main_skill():
    await skillsnetwork.prepare(url=dataset_url, path=data_dir, overwrite=True)
async def main_tar():
    await download_tar_dataset(dataset_url, tar_path, data_dir)
async def main():
    await download_model(pytorch_state_dict_url, pytorch_state_dict_path)


try:
    check_skillnetwork_extraction(data_dir)
    if __name__ == "__main_skill__":
        asyncio.run(main_skill())
except Exception as e:
    print(e)
    print("Primary download/extraction method failed.")
    print("Falling back to manual download and extraction...")
    import tarfile
    import httpx
    from pathlib import Path
    file_name = Path(dataset_url).name
    tar_path = os.path.join(data_dir, file_name)
    if __name__ == "__main_tar__":
        asyncio.run(main_tar())


pytorch_state_dict_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rFBrDlu1NNcAzir5Uww8eg/pytorch-cnn-vit-ai-capstone-model-state-dict.pth"
pytorch_state_dict_name = "pytorch_cnn_vit_ai_capstone_model_state_dict.pth"
pytorch_state_dict_path = os.path.join(data_dir, pytorch_state_dict_name)

if __name__ == "__main__":
    asyncio.run(main())

