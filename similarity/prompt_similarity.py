import torch
import clip

def compute_text_embeddings(prompts):
    # 加载 CLIP 模型和预处理器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("/home/jovyan/models/clip-ViT-B-32/models/snapshots/11fb331c2c388748c110926aa8013161cb5a85b5/", device=device)

    # 处理文本提示
    text_inputs = clip.tokenize(prompts).to(device)

    # 计算文本的嵌入
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # 归一化嵌入
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

def cosine_similarity(embedding1, embedding2):
    # 计算余弦相似度
    similarity = torch.matmul(embedding1, embedding2.T)
    return similarity.item()

def main():
    # 示例文本提示
    prompt1 = "a photo of a cat"
    prompt2 = "a photo of a dog"

    # 计算文本提示的嵌入
    embeddings = compute_text_embeddings([prompt1, prompt2])
    
    # 计算两个文本提示的余弦相似度
    similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])

    # 打印结果
    print(f"Cosine similarity between '{prompt1}' and '{prompt2}': {similarity}")

if __name__ == "__main__":
    main()
