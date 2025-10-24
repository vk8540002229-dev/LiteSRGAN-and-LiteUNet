from Dataloader import DataLoader
from LiteSRGAN import LiteSRGAN, LiteSRGAN_engine
import argparse
import tensorflow as tf
import os
from tqdm import tqdm

# -----------------------------
# Argument Parser (unchanged)
# -----------------------------
parser = argparse.ArgumentParser(description='light-SRGAN training script.')

parser.add_argument('--images_dir', default=r"/content/drive/MyDrive/images_001/original_images",
                    type=str, help='provide a path containing all the data')
parser.add_argument('--img_width', default=128, type=int)
parser.add_argument('--img_height', default=128, type=int)
parser.add_argument('--upsampling_blocks', default=1, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--decay_steps', default=50000, type=int)
parser.add_argument('--decay_rate', default=0.1, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--pretraining_epochs', default=9, type=int)
parser.add_argument('--generator_weights', default=None, type=str)
parser.add_argument('--discriminator_weights', default=None, type=str)

args = parser.parse_args()

# -----------------------------
# Google Drive Paths
# -----------------------------
BASE_DIR = '/content/drive/MyDrive/LiteSRGAN_data'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
GEN_DIR = os.path.join(BASE_DIR, 'generatedTrails')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
PRETRAIN_DIR = os.path.join(CKPT_DIR, 'pretrain')
TRAIN_DIR = os.path.join(CKPT_DIR, 'training')

for d in [BASE_DIR, MODELS_DIR, GEN_DIR, CKPT_DIR, PRETRAIN_DIR, TRAIN_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"✅ All training data will be stored under: {BASE_DIR}")

# -----------------------------
# Helper: find latest checkpoint by filename
# -----------------------------
def get_latest_checkpoint(dir_path, prefix):
    ckpts = [f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith('.weights.h5')]
    if not ckpts:
        return None, -1
    ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest = ckpts[-1]
    epoch_num = int(latest.split('_')[-1].split('.')[0])
    return os.path.join(dir_path, latest), epoch_num

# -----------------------------
# Initialize DataLoader and Model
# -----------------------------
dl = DataLoader(args)
datagen = dl.dataGenerator()

lite_SRGAN = LiteSRGAN(args)
lite_SRGAN_engine = LiteSRGAN_engine(args, lite_SRGAN)
print("last layer shape of discriminator: ", lite_SRGAN.discriminator.output_shape)

# =========================================================
# === PRETRAINING PHASE (resumable .weights.h5 checkpoints)
# =========================================================
latest_pre_ckpt, start_pre_epoch = get_latest_checkpoint(PRETRAIN_DIR, "pretraining_")

if latest_pre_ckpt:
    lite_SRGAN.generator.load_weights(latest_pre_ckpt)
    print(f"🔁 Resumed generator pretraining from {latest_pre_ckpt} (epoch {start_pre_epoch})")
else:
    print("🚀 Starting generator pretraining from scratch...")
    start_pre_epoch = -1  # No checkpoint yet

for i in range(start_pre_epoch + 1, args.pretraining_epochs):
    print(f"------------- Pre-training epoch {i} -------------")
    lite_SRGAN_engine.generator_pretraining(datagen, i)

    # save weights in TF3-compatible format
    save_path = os.path.join(PRETRAIN_DIR, f"pretraining_{i}.weights.h5")
    lite_SRGAN.generator.save_weights(save_path)
    print(f"💾 Saved pretraining checkpoint: {save_path}")

print("------------- End of generator pre-training -------------")

# =========================================================
# === MAIN TRAINING PHASE (resumable .weights.h5 checkpoints)
# =========================================================
latest_train_gen, start_train_epoch = get_latest_checkpoint(TRAIN_DIR, "training_")
latest_train_disc, _ = get_latest_checkpoint(TRAIN_DIR, "training_")

if latest_train_gen and latest_train_disc:
    lite_SRGAN.generator.load_weights(latest_train_gen)
    lite_SRGAN.discriminator.load_weights(latest_train_disc)
    print(f"✅ Resumed full training from {latest_train_gen} (epoch {start_train_epoch})")
else:
    print("⚙️ Starting full SRGAN training from scratch...")
    start_train_epoch = -1  # No checkpoint yet

# -----------------------------
# Optimizers (fallback if missing)
# -----------------------------
try:
    g_opt = lite_SRGAN_engine.g_optimizer
except AttributeError:
    g_opt = tf.keras.optimizers.Adam(args.lr)
try:
    d_opt = lite_SRGAN_engine.d_optimizer
except AttributeError:
    d_opt = tf.keras.optimizers.Adam(args.lr)

# -----------------------------
# Training Loop
# -----------------------------
for i in range(start_train_epoch + 1, args.epochs):
    print(f"============= SRGAN Training Epoch {i} =============")
    datagen = dl.dataGenerator()
    lite_SRGAN_engine.train(datagen, 100, i)
    lite_SRGAN_engine.saveTrails(4, i)

    # save model weights
    gen_path = os.path.join(MODELS_DIR, f'generator_epoch_{i+1}.weights.h5')
    disc_path = os.path.join(MODELS_DIR, f'discriminator_epoch_{i+1}.weights.h5')
    lite_SRGAN.generator.save_weights(gen_path)
    lite_SRGAN.discriminator.save_weights(disc_path)
    print(f"💾 Saved model weights for epoch {i+1}")

    # save training checkpoints (so resume works automatically)
    gen_ckpt_path = os.path.join(TRAIN_DIR, f"training_{i}_generator.weights.h5")
    disc_ckpt_path = os.path.join(TRAIN_DIR, f"training_{i}_discriminator.weights.h5")
    lite_SRGAN.generator.save_weights(gen_ckpt_path)
    lite_SRGAN.discriminator.save_weights(disc_ckpt_path)
    print(f"📍 Saved training checkpoint after epoch {i+1}")

print("🎉 Training completed and all data saved to Google Drive.")
