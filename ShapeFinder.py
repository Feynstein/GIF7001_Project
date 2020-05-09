'''
Par:
Jean-Christophe Ruel

Novembre 2018, Universite Laval
Dans le cadre du cours GIF-7001
'''

import cv2
import numpy as np
from utils.graphics import pltshow
from utils.img_handle import GetImg, rotate_image
from utils.shapes import FindShapes
import time
from scipy.optimize import minimize, minimize_scalar


def searchformatch(angle, template, scene, optimize=False, maskinside=False):
    # Il faut maximiser res en jouant sur l'angle
    method_used = 4
    methods = {0: ('cv2.TM_CCOEFF', 'max'), 1: ('cv2.TM_CCOEFF_NORMED', 'max'), 2: ('cv2.TM_CCORR', 'max'),
               3: ('cv2.TM_CCORR_NORMED', 'max'), 4: ('cv2.TM_SQDIFF', 'min'), 5: ('cv2.TM_SQDIFF_NORMED', 'min')}
    method, extrema = methods[method_used]
    rotated_template = rotate_image(template, angle)

    mask_inside = None
    ret, mask_edges = cv2.threshold(rotated_template, 10, 255, cv2.THRESH_BINARY)
    if maskinside:
        mask_inside = mask_edges.copy()
        cv2.floodFill(mask_inside, None, (0, 0), 255)
        mask_inside = cv2.bitwise_not(mask_inside)
        mask = cv2.add(mask_edges, mask_inside)
    else:
        mask = mask_edges

    res = cv2.matchTemplate(scene, rotated_template, method=eval(method), mask=mask)


    if optimize:
        return eval('res.'+extrema+'()')
    else:
        # print('norm: {}'.format(1-norm))
        norm, pt = normsqdiff(rotated_template, scene, mask, res, extrema=extrema)
        showDict = {'mask': mask, 'mask_edges': mask_edges, 'mask_inside': mask_inside}
        mask_dict = {key: m for key, m in showDict.items() if m is not None}
        return 1-norm, rotated_template, mask_dict, pt

def normsqdiff(template, scene, mask, res, extrema='min'):
    '''

    :param template: image template
    :param scene: image scene
    :param mask: mask du template
    :param res: resultat du template matching en faisant l'usage de la methode cv2.TM_SQDIFF non-normalisee
    :param extrema: Pour la methode cv2.TM_SQDIFF, la valeur min correspond au meilleur match
    :return: la normalisation du meilleur match et sa localisation

    TODO:
    Cette focntion de normalisation serait beaucoup plus rapide si les operations pouvaient
    etre tranferes dans l'espace fourier. http://www.jot.fm/issues/issue_2010_03/column2.pdf

    '''

    w, h = template.shape[:2]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    pt_dict = {'max':max_loc, 'min':min_loc}
    pt = pt_dict[extrema][::-1]

    scene = scene[pt[0]:pt[0] + w, pt[1]:pt[1] + h][mask > 0].astype(float)
    template = template[mask > 0].astype(float)
    template = cv2.normalize(template, template, alpha= -1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    scene = cv2.normalize(scene, scene, alpha= -1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    sqsum_T = np.sum((template)**2)
    sqsum_I = np.sum((scene)**2)
    sqdiff = np.sum((template-scene)**2)
    normed = sqdiff/(np.sqrt(sqsum_I*sqsum_T)+1e-16)
    return normed, pt


def templateMatching(scene_name, template_name, threshold, res_ratio, maskinside=False, optimize=False):

    # Definition de l'emplacement des images
    path_scene = './/images//' + scene_name
    path_template = './/images//' + template_name
    img_scene = GetImg(path_scene)
    img_template = GetImg(path_template)

    # Resize pour limiter le temps de calcul
    img_scene.resize(ratio=res_ratio) #1 VS 1/4
    img_template.resize(ratio=res_ratio) #1 VS 1/4

    # Crop de l'image template selon la definition de l'usager
    img_template.cropNsave(save=False)

    #Debut du decompte
    start_time = time.time()

    # Transfert de l'image dans le manipulateur Findshapes
    scene = FindShapes(img_scene())
    scene.find_edges()

    template = FindShapes(img_template.crop)
    template.find_edges()

    # Spread les edges avec un gradient afin d'augmenter l'erreur admissible a la perspertive et
    # de faciliter l'optimisation locale
    template.smooth_edges(spread_value=200)
    scene.smooth_edges(spread_value=30)
    print('Scene shape: {}, Template shape: {}'.format(scene.img.shape, template.img.shape))

    angle_range = np.linspace(0, 360, 36)
    #np.random.shuffle(angle_range)
    print('Number of Bruteforce iterations: {}'.format(angle_range.shape[0]))

    bruteforce = {searchformatch(angle, template.smooth, scene.smooth, optimize=True): angle for angle in angle_range}
    init_angle = bruteforce[min(bruteforce)]

    #Ne pas utiliser minimize avec la methode 'SLSQP'
    if optimize:
        # Avec optimize = True, en plus de realiser le brute force sur les rotation de l'image,
        # on optimise son orientation avec un optimizer. Ca permet de gagner quelque degres de precision
        # mais alourdit le calcul.
        rep = minimize(searchformatch, x0=init_angle, args=(template.smooth, scene.smooth, True, maskinside), tol=1e-4,
                       method='Nelder-Mead', options={'fatol' :1e-5, 'maxiter': 70})
        Final_angle = rep.x
        #rep = minimize_scalar(searchformatch, method='brent', args=(template.smooth, scene.smooth, True), tol=1e-12)
        norm, rotated_template, mask_dict, pt = searchformatch(Final_angle, template.smooth, scene.smooth,
                                                          maskinside=maskinside)
        print('Initial angle: {}, Final angle: {}'.format(init_angle, Final_angle))
    else:
        Final_angle = init_angle
        norm, rotated_template, mask_dict, pt = searchformatch(Final_angle, template.smooth, scene.smooth,
                                                          maskinside=maskinside)
    print('Match confidence: {0:.2f} '.format(norm))

    templateMatch = None
    edges = rotate_image(template.edges, Final_angle)
    if norm > threshold:
        # On superpose les edges du template (en bleu) sur l'image scene a
        # l'endroit ou le match a eu lieu
        w, h = mask_dict['mask'].shape[:2]
        templateMatch = scene.img.copy()
        match = np.zeros(templateMatch.shape)
        match[pt[0]: pt[0] + w, pt[1]:pt[1] + h, :][edges>0] = [0, 102, 255]
        templateMatch[pt[0]: pt[0] + w, pt[1]:pt[1] + h, :][edges>0] = [0, 0, 0]
        templateMatch = cv2.add(templateMatch.astype(np.uint8), match.astype(np.uint8))
    else:
        print('Aucun match trouve')

    print('time = %.2f' % (time.time() - start_time))
    showDict = {1: (templateMatch, 'Result'), 2: (scene.smooth, 'Scene Edges'), 3: (template.smooth, 'Template smooth edges'),
                4: (rotated_template, 'Rotated template'), 5: (mask_dict['mask'], 'Mask alpha')}
    images_fig = tuple([image for key, image in showDict.items() if image[0] is not None])
    pltshow(images_fig)


if __name__ == '__main__':
    # Pour plus de precision, imposer optimize=True. Pour encore plus de precision, choisir un template
    # qui a la meme perspective que l'objet dans la scene, idealement: les objets devraient etre situes
    # le long de l'axe z de la camera. Il est aussi preferable d'utiliser la resolution la plus grande possible
    templateMatching(scene_name='TP001_H1_130_0007.tif', template_name='TP001_H1_060_0003.tif', threshold=0.3, res_ratio=1/6, maskinside=True, optimize=True)

