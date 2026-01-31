dataset_path = os.path.join(data_dir, "images_dataSAT")

img_w, img_h = 64, 64
batch_size = 128
num_classes = 2
agri_class_labels = ["non-agri", "agri"]

depth = 3
attn_heads = 6
embed_dim = 768
