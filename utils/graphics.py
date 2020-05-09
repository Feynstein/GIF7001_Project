import matplotlib.pyplot as plt


def pltshow(images, figsize=None, cmap=None, width=3):

    if type(images) is not tuple:
        plt.figure(figsize=figsize)
        plt.imshow(images, cmap=cmap, interpolation='nearest')
        plt.tight_layout()
        plt.axis('off')
        plt.show()
    else:
        if len(images) < 3:
            width = len(images)
        else:
            pass
        height = round(len(images) / width + 0.5)
        fig, axes = plt.subplots(nrows=height, ncols=width, figsize=(width*5, height*4))

        # Gère l'affichage en fonction du nombre d'images disponibles:
        for i, im in enumerate(axes.flat):
            if i < len(images):
                im.imshow(images[i][0], cmap='gray')
                im.set_xticks([])
                im.set_yticks([])
                im.set_title(images[i][1])
                im.axis('off')
            else:
                im.set_visible(False)
        plt.tight_layout()
        plt.show()

