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

# Dataloader args
parser.add_argument('--images_dir', default=r"/content/drive/MyDrive/images_001/original_images",
                    type=str, help='provide a path containing all the data')
parser.add_argument('--img_width', default=128, type=int)
parser.add_argument('--img_height', default=128, type=int)

# Model args
parser.add_argument('--upsampling_blocks', default=1, type=int,
                    help='Number of upsampling blocks for upscaling.')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--decay_steps', default=50000, type=int)
parser.add_argument('--decay_rate', default=0.1, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--pretraining_epochs', default=100, type=int)
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

for d in [BASE_DIR, MODELS_DIR, GEN_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"✅ All training data will be stored under: {BASE_DIR}")

# -----------------------------
# Initialize DataLoader
# -----------------------------
dl = DataLoader(args)
datagen = dl.dataGenerator()

# -----------------------------
# Initialize Model
# -----------------------------
lite_SRGAN = LiteSRGAN(args)
lite_SRGAN_engine = LiteSRGAN_engine(args, lite_SRGAN)
print("last layer shape of discriminator: ", lite_SRGAN.discriminator.output_shape)

# -----------------------------
# Pretraining Checkpoint
# -----------------------------
pre_ckpt = tf.train.Checkpoint(generator=lite_SRGAN.generator)
pre_ckpt_manager = tf.train.CheckpointManager(pre_ckpt, os.path.join(CKPT_DIR, 'pretrain'), max_to_keep=3)

if pre_ckpt_manager.latest_checkpoint:
    pre_ckpt.restore(pre_ckpt_manager.latest_checkpoint)
    print(f"🔁 Resumed generator pretraining from {pre_ckpt_manager.latest_checkpoint}")
else:
    print("🚀 Starting generator pretraining from scratch...")

# -----------------------------
# Pretraining Loop
# -----------------------------
for i in range(args.pretraining_epochs):
    lite_SRGAN_engine.generator_pretraining(datagen, i)
    pre_ckpt_manager.save()
    print(f"💾 Saved pretraining checkpoint after epoch {i+1}")

print("------------- End of generator pre-training -------------")

# -----------------------------
# Main Training Checkpoint
# -----------------------------
try:
    g_opt = lite_SRGAN_engine.g_optimizer
except AttributeError:
    g_opt = tf.keras.optimizers.Adam(args.lr)
try:
    d_opt = lite_SRGAN_engine.d_optimizer
except AttributeError:
    d_opt = tf.keras.optimizers.Adam(args.lr)

train_ckpt_dict = {
    'generator': lite_SRGAN.generator,
    'discriminator': lite_SRGAN.discriminator
}
if g_opt is not None:
    train_ckpt_dict['g_optimizer'] = g_opt
if d_opt is not None:
    train_ckpt_dict['d_optimizer'] = d_opt

train_ckpt = tf.train.Checkpoint(**train_ckpt_dict)
train_ckpt_manager = tf.train.CheckpointManager(train_ckpt, os.path.join(CKPT_DIR, 'training'), max_to_keep=3)

if train_ckpt_manager.latest_checkpoint:
    train_ckpt.restore(train_ckpt_manager.latest_checkpoint)
    print(f"✅ Resumed training from checkpoint: {train_ckpt_manager.latest_checkpoint}")
else:
    print("⚙️ Starting full SRGAN training from scratch...")

# -----------------------------
# Training Loop
# -----------------------------
for i in range(args.epochs):
    datagen = dl.dataGenerator()
    lite_SRGAN_engine.train(datagen, 100, i)
    lite_SRGAN_engine.saveTrails(4, i)

    # Save model weights
    gen_path = os.path.join(MODELS_DIR, f'generator_epoch_{i+1}.h5')
    disc_path = os.path.join(MODELS_DIR, f'discriminator_epoch_{i+1}.h5')
    lite_SRGAN.generator.save_weights(gen_path)
    lite_SRGAN.discriminator.save_weights(disc_path)
    print(f"💾 Saved generator/discriminator weights for epoch {i+1}")

    # Save checkpoint (includes optimizers)
    train_ckpt_manager.save()
    print(f"📍 Training checkpoint saved after epoch {i+1}")

print("🎉 Training completed and all data saved to Google Drive.")
