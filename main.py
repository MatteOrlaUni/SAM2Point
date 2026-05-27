# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import open3d as o3d
import laspy
import cv2
from scipy.spatial import KDTree

from segment import seg_point
import sam2point.dataset as dataset
from sam2point.voxelizer import Voxelizer
from sam2point.utils import cal
from show import render_scene, render_scene_outdoor

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

punto_selezionato_reale = None


def apri_nuvola_las(percorso_las):
    las = laspy.read(percorso_las)
    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    z = np.array(las.z, dtype=np.float32)
    punti_reali = np.vstack((x, y, z)).T

    print(f"Punti totali nel file LAS: {punti_reali.shape[0]}")

    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        r = np.array(las.red, dtype=np.float32)
        g = np.array(las.green, dtype=np.float32)
        b = np.array(las.blue, dtype=np.float32)
        colori_las = np.vstack((r, g, b)).T
        if colori_las.max() > 1.0:
            divisore = 65535.0 if colori_las.max() > 255 else 255.0
            colori_las /= divisore
    else:
        print("[WARN] Il file LAS non contiene informazioni sul colore. Genero colore grigio di default.")
        colori_las = np.ones_like(punti_reali) * 0.5

    return punti_reali, colori_las


def seleziona_punto_interattivo(punti_reali, risoluzione_pixel=800):
    """Genera l'interfaccia di selezione usando una normalizzazione isotropa temporanea."""
    global punto_selezionato_reale
    punto_selezionato_reale = None 

    x = punti_reali[:, 0]
    y = punti_reali[:, 1]
    z = punti_reali[:, 2]

    # Indicizzazione basata sulle coordinate reali per non perdere precisione
    print("Indicizzazione della nuvola di punti...")
    kdtree = KDTree(punti_reali[:, :2])

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    delta_x = x_max - x_min
    delta_y = y_max - y_min

    if delta_x > delta_y:
        larghezza_img = risoluzione_pixel
        altezza_img = int(risoluzione_pixel * (delta_y / delta_x))
    else:
        altezza_img = risoluzione_pixel
        larghezza_img = int(risoluzione_pixel * (delta_x / delta_y))

    larghezza_img = max(1, larghezza_img)
    altezza_img = max(1, altezza_img)

    u_pixel = ((x - x_min) / delta_x * (larghezza_img - 1)).astype(np.int32) if delta_x != 0 else np.zeros_like(x, dtype=np.int32)
    v_pixel = ((y_max - y) / delta_y * (altezza_img - 1)).astype(np.int32) if delta_y != 0 else np.zeros_like(y, dtype=np.int32)

    z_norm = ((z - z_min) / (z_max - z_min) * 255).astype(np.uint8) if z_max != z_min else np.zeros_like(z, dtype=np.uint8)
    immagine_2d = np.zeros((altezza_img, larghezza_img, 3), dtype=np.uint8)
    indici_ordinati = np.argsort(z_norm)
    colori_mappati = cv2.applyColorMap(z_norm, cv2.COLORMAP_JET).squeeze()
    immagine_2d[v_pixel[indici_ordinati], u_pixel[indici_ordinati]] = colori_mappati[indici_ordinati]

    info_proiezione = {
        'x_min': x_min, 'y_max': y_max,
        'delta_x': delta_x, 'delta_y': delta_y,
        'larghezza_img': larghezza_img, 'altezza_img': altezza_img,
        'kdtree': kdtree, 'punti_reali': punti_reali,
        'immagine': immagine_2d
    }

    nome_finestra = "Seleziona Punto Prompt (Mappa di Calore Z)"
    cv2.namedWindow(nome_finestra)
    cv2.setMouseCallback(nome_finestra, gestisci_click, param=info_proiezione)

    print("\n[INFO] Interfaccia Pronta.")
    print("1. Fai CLICK SINISTRO sulla mappa per posizionare il prompt.")
    print("2. Premi INVIO o ESC per confermare il punto selezionato.")

    while True:
        cv2.imshow(nome_finestra, info_proiezione['immagine'])
        tasto = cv2.waitKey(10) & 0xFF
        if tasto == 13 or tasto == 27:
            break
        if cv2.getWindowProperty(nome_finestra, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return punto_selezionato_reale


def gestisci_click(event, u, v, flags, param):
    global punto_selezionato_reale
    
    if event == cv2.EVENT_LBUTTONDOWN:
        x_min = param['x_min']
        y_max = param['y_max']
        delta_x = param['delta_x']
        delta_y = param['delta_y']
        larghezza_img = param['larghezza_img']
        altezza_img = param['altezza_img']
        kdtree = param['kdtree']
        punti_reali = param['punti_reali']
        immagine_visualizzata = param['immagine']

        x_stimata = x_min + (u / (larghezza_img - 1)) * delta_x
        y_stimata = y_max - (v / (altezza_img - 1)) * delta_y

        _, indice_vicino = kdtree.query([x_stimata, y_stimata])
        punto_selezionato_reale = punti_reali[indice_vicino]

        img_feedback = immagine_visualizzata.copy()
        cv2.circle(img_feedback, (u, v), 5, (255, 255, 255), -1)
        cv2.circle(img_feedback, (u, v), 6, (0, 0, 0), 1)
        param['immagine'] = img_feedback

        print(f"\n[OK] Prompt acquisito nelle coordinate reali della nuvola:")
        print(f"-> X: {punto_selezionato_reale[0]:.4f}, Y: {punto_selezionato_reale[1]:.4f}, Z: {punto_selezionato_reale[2]:.4f}")


def main():
    class Args:
        file_name = "skateboard.las"  
        mode = "nearest"
        theta = 0.5
        voxel_size = 0.05  # Nelle scene outdoor, un voxel di 5-10 cm è lo standard
        dataset = "KITTI"  # CAMBIATO DA Objaverse A KITTI
        prompt_type = "point"
        sample_idx = 0
        prompt_idx = 0

    args = Args()
    percorso_completo = os.path.join("inputs", args.file_name)

    if not os.path.exists(percorso_completo):
        print(f"[ERROR] Inserisci il file {args.file_name} dentro la cartella 'inputs/'")
        return

    # 1. Caricamento nativo senza alterazioni geometriche
    point, color = apri_nuvola_las(percorso_completo)
    
    # 2. Interfaccia utente per estrarre le coordinate reali del prompt
    sel_point = seleziona_punto_interattivo(point)
    if sel_point is None:
        print("[ERRORE] Elaborazione annullata: nessun punto selezionato.")
        return

    # Formattazione del prompt come richiesto dall'algoritmo aziendale
    info_point_prompts = [sel_point.tolist()]
    prompt_point = list(info_point_prompts[0])
    prompt_point = [[0.5527776, 0.7294311, 0.685305 ]]
    prompt_point = sel_point.tolist()
    prompt_box = None

    # Da qui in poi il codice esegue il flusso IDENTICO all'originale dell'azienda
    print("\n[INFO] Avvio voxelizzazione ed inferenza SAM2Point...")
    point_color = np.concatenate([point, color], axis=1)
    voxelizer = Voxelizer(voxel_size=args.voxel_size, clip_bound=None)
    
    labels_in = point[:, :1].astype(int)
    locs, feats, labels, inds_reconstruct = voxelizer.voxelize(point, color, labels_in)

    # Esecuzione della segmentazione con il punto reale estratto
    mask = seg_point(locs, feats, info_point_prompts, args)
    
    point_locs = locs[inds_reconstruct]
    point_mask = mask[point_locs[:, 0], point_locs[:, 1], point_locs[:, 2]]
    
    point_mask = point_mask.unsqueeze(-1)
    point_mask_not = ~point_mask
    
    point, color = point_color[:, :3], point_color[:, 3:]
    new_color = color * point_mask_not.numpy() + (color * 0 + np.array([[0., 1., 0.]])) * point_mask.numpy()

    os.makedirs('results', exist_ok=True)
    base_name = os.path.splitext(args.file_name)[0]
    name = f"user_{base_name}"

    # Rendering standard della scena
    render_scene_outdoor(point, new_color, name, prompt_point=prompt_point, prompt_box=prompt_box)
    print(f"\n[COMPLETATO] Elaborazione terminata. Risultato salvato in 'results/{name}'")


if __name__ == "__main__":
    main()