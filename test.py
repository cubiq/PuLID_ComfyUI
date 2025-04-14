import torch
from nodes import (
    CheckpointLoaderSimple,
    KSampler,
    EmptyLatentImage,
    CLIPTextEncode,
    VAEDecode,
    LoadImage,
    SaveImage,
)
from .pulid import (
    PulidModelLoader,
    PulidInsightFaceLoader,
    PulidEvaClipLoader,
    ApplyPulid,
)


@torch.no_grad
def test_sdxl_pulid_v1():
    model, clip, vae = CheckpointLoaderSimple().load_checkpoint(
        "sd_xl_base_1.0.safetensors"
    )
    assert model is not None
    # pulid = PulidModelLoader().load_model("ip-adapter_pulid_sdxl_fp16.safetensors")[0]
    pulid = PulidModelLoader().load_model("ip-adapter_pulidv1.1_sdxl_fp16.safetensors")[
        0
    ]
    face_analysis = PulidInsightFaceLoader().load_insightface("CUDA")[0]
    eva_clip = PulidEvaClipLoader().load_eva_clip()[0]

    image = LoadImage().load_image("ldh.png")[0]

    workmodel = ApplyPulid().apply_pulid(
        model,
        pulid,
        eva_clip,
        face_analysis,
        image,
        method="fidelity",
        weight=1.0,
        start_at=0.0,
        end_at=1.0,
    )[0]

    latent = EmptyLatentImage().generate(1024, 1024)[0]
    positive = CLIPTextEncode().encode(
        clip, "portrait of a man, detailed, realistic, cinematic"
    )[0]
    negative = CLIPTextEncode().encode(
        clip,
        "painting, drawing, anime, worst quality, low quality, blurry, black and white",
    )[0]
    latent = KSampler().sample(
        workmodel, 42, 15, 5.2, "dpmpp_sde", "karras", positive, negative, latent
    )[0]

    image_tensor = VAEDecode().decode(vae, latent)[0]
    ret = SaveImage().save_images(image_tensor, filename_prefix="test")
    print(ret)
    assert ret["ui"]["images"] is not None


def pulid2ipa(input_path, output_path):
    from safetensors.torch import load_file, save_file

    sd_pulid = load_file(input_path)
    sd_ipa = {}
    for k, v in sd_pulid.items():
        nk = (
            k.replace("id_adapter.", "image_proj.")
            .replace("id_adapter_attn_layers.", "ip_adapter.")
            .replace("id_to_k.", "to_k_ip.")
            .replace("id_to_v.", "to_v_ip.")
        )
        sd_ipa[nk] = v.to(torch.float16)

    save_file(sd_ipa, output_path)


if __name__ == "__main__":
    test_sdxl_pulid_v1()
