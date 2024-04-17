import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def show(r_img, s_img, c_limage, fake_img):
    fig, axes = plt.subplots(1, 4, figsize=(10, 7))

    axes[0].imshow(r_img)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(s_img)
    axes[1].set_title('Sketch')
    axes[1].axis('off')

    axes[2].imshow(c_limage)
    axes[2].set_title('Fake Sketch')
    axes[2].axis('off')

    axes[3].imshow(fake_img)
    axes[3].set_title('Fake Image')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()
