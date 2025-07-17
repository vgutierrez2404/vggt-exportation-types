from typing import Union

import cv2
import os
import numpy as np
from PIL import Image
import shutil

def resize_images_exif_info(input_folder:str, output_folder:str, selected_frames:list[str] = [], resample_factor:int = 0.5, exif_info:bool = False) -> None:
    """Takes the list of frames that will be used in the reconstruction and 
    resize them. 
    """

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of files in the input folder
    files = os.listdir(input_folder)
    if selected_frames:
        # This selects only the frames that has been selected. 
        files = list(set(files) & set(selected_frames))

    for file in files:
        # Get the full path of the original image
        original_path = os.path.join(input_folder, file)

        # Open the image using Pillow
        original_image = Image.open(original_path)

        # Resize the image using resampling
        new_height = int(original_image.height * resample_factor)
        new_width = int(original_image.width * resample_factor)
        resized_image = original_image.resize((new_width, new_height), Image.LANCZOS) # LANCZOS is the filter thats been used for rescaling the image. 

        # Create the path for the resized image in the output folder
        resized_path = os.path.join(output_folder, file)

        # Preserve the EXIF data
        if exif_info:
            exif_data = original_image.info.get('exif')
            resized_image.save(resized_path, exif=exif_data)
            
        else:
            resized_image.save(resized_path)



def crop_image_into_patches(image_path, output_directory, patch_size=512):
    # Load the original image
    original_image = cv2.imread(image_path)

    # Get dimensions of the original image
    height, width, _ = original_image.shape

    # Calculate the number of patches in each dimension
    num_patches_horizontal = -(-width // patch_size)  # Ceiling division
    num_patches_vertical = -(-height // patch_size)  # Ceiling division

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over patches
    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            # Calculate patch coordinates
            y_start = i * patch_size
            y_end = min((i + 1) * patch_size, height)
            x_start = j * patch_size
            x_end = min((j + 1) * patch_size, width)

            # Crop the patch from the original image
            patch = original_image[y_start:y_end, x_start:x_end]

            # Create a patch filled with zeros if the patch is smaller than 512x512
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                zeros_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                zeros_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = zeros_patch

            # Save the patch with a filename starting with the original image name
            patch_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_patch_{i}_{j}.jpg"
            patch_filepath = os.path.join(output_directory, patch_filename)
            cv2.imwrite(patch_filepath, patch)

def reconstruct_image_from_patches(base_image_name, input_directory, output_directory, patch_size=512, original_size=(1080,1920,3),GrayScale=False):

    # List all patch files with the given base image name
    patch_files = [f for f in os.listdir(input_directory) if f.startswith(base_image_name + "_patch")]

    # Sort patch files based on their indices
    patch_files.sort(key=lambda x: (int(x.split('_')[3]), int(os.path.splitext(x.split('_')[4])[0])))

    # Initialize variables to store the reconstructed image
    reconstructed_image = np.zeros(((int(patch_files[-1].split('_')[3])+1)*patch_size, (int((patch_files[-1].split('_')[4])[0])+1)*patch_size, 3), dtype=np.uint8)
    current_row = 0
    current_col = 0

    # Iterate over patch files
    for ii, patch_file in enumerate(patch_files):
        #print(ii)
        #print(patch_file)
        # Load the patch image
        patch_path = os.path.join(input_directory, patch_file)
        patch = cv2.imread(patch_path)

        # Update row and column indices
        current_row, current_col = int(patch_file.split('_')[3]), int(patch_file.split('_')[4].split('.')[0])

        # Check if this is the first patch
        #if reconstructed_image is None:
        #    reconstructed_image = np.zeros((patch_size * (current_row + 1), patch_size * (current_col + 1), 3), dtype=np.uint8)

        # Place the patch in the corresponding position
        reconstructed_image[current_row * patch_size:(current_row + 1) * patch_size,
                            current_col * patch_size:(current_col + 1) * patch_size] = patch

    # Save the reconstructed image
    reconstructed_image = reconstructed_image[:original_size[0], :original_size[1]]
    reconstructed_filename = base_image_name + ".jpg"
    os.makedirs(output_directory, exist_ok=True)
    reconstructed_filepath = os.path.join(output_directory, reconstructed_filename)

    if GrayScale:
        gray_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(reconstructed_filepath, gray_image)
    else:
        cv2.imwrite(reconstructed_filepath, reconstructed_image)


def process_images_and_create_patches(input_folder, output_folder_patches, patch_size=512):
    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Iterate over each image file
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        # Process the image and create patches
        crop_image_into_patches(image_path, output_folder_patches, patch_size)

def reconstruct_all_images_from_patches(input_directory_patches, output_directory_reconstructed, patch_size=512, original_size=(1080,1920,3),GrayScale=False):
    # List all patch files in the input directory
    patch_files = [f for f in os.listdir(input_directory_patches) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Extract unique basenames from patch files
    unique_basenames = set([os.path.splitext(f.split('_patch')[0])[0] for f in patch_files])

    # Iterate over each unique basename
    for base_image_name in unique_basenames:
        #input_directory = input_directory_patches
        #output_directory = output_directory_reconstructed
        # Reconstruct image from patches for the current basename
        reconstruct_image_from_patches(base_image_name, input_directory_patches, output_directory_reconstructed, patch_size, original_size, GrayScale)

def divide_image_into_patches_overlap(image_path, output_directory, patch_size=512, overlap_percentage=50):
    # Load the original image
    original_image = cv2.imread(image_path)
    os.makedirs(output_directory, exist_ok=True)
    # Get dimensions of the original image
    height, width, _ = original_image.shape


    overlap_pixels = int(patch_size * overlap_percentage / 100)
    print(overlap_pixels)

    for y_start in range(0, height, patch_size - overlap_pixels):
        for x_start in range(0, width, patch_size - overlap_pixels):

            y_end = min(y_start + patch_size, height)
            x_end = min(x_start + patch_size, width)

            patch = original_image[y_start:y_end, x_start:x_end]
            if (patch.shape[0] > overlap_pixels) and (patch.shape[1] > overlap_pixels):
                if (patch.shape[0] < patch_size) or (patch.shape[1] < patch_size):
                    print(patch.shape)
                    patchPadding = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    patchPadding[0:patch.shape[0], 0:patch.shape[1], :] = patch
                    patchPadding[0:patch.shape[0], patch.shape[1]:patch_size, :] = patch[0:patch.shape[0], patch.shape[1]-1:2*patch.shape[1]-patch_size-1:-1, :]
                    patchPadding[patch.shape[0]:patch_size, 0:patch.shape[1], :] = patch[patch.shape[0] - 1:2 * patch.shape[0] - patch_size - 1:-1, 0:patch.shape[1], :]
                    patchPadding[patch.shape[0]:patch_size, patch.shape[1]:patch_size, :] = patch[patch.shape[0] - 1:2 * patch.shape[0] - patch_size - 1:-1,
                                                                                            patch.shape[1]-1:2*patch.shape[1]-patch_size-1:-1, :]

                    patch = patchPadding

                patch_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_patch_{y_start}_{x_start}.jpeg"
                patch_filepath = os.path.join(output_directory, patch_filename)
                cv2.imwrite(patch_filepath, patch)

def process_images_and_create_patches_overlap(input_folder, output_folder_patches, patch_size=512, overlap_percentage=50):
    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Iterate over each image file
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        # Process the image and create patches
        divide_image_into_patches_overlap(image_path, output_folder_patches, patch_size, overlap_percentage)

def reconstruct_image_from_patches_overlap(base_image_name, input_directory, output_filepath, weight_mask, patch_size=512, original_size=(1080, 1920, 3), GrayScale=False):
    # List all patch files in the input directory
    patch_files = [f for f in os.listdir(input_directory) if f.startswith(base_image_name+"_patch_")]

    # Sort patch files based on their coordinates
    patch_files.sort(key=lambda x: (int(x.split('_')[-2]), int((x.split('_')[-1]).split('.')[0])))

    image_shape = np.asarray([int(patch_files[-1].split('_')[-2]), int((patch_files[-1].split('_')[-1]).split('.')[0])]) + patch_size

    # Inicializar la imagen reconstruida
    reconstructed_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.float32)
    weights = np.zeros_like(reconstructed_image, dtype=np.float32)

    # Iterar sobre los parches y reconstruir la imagen
    for patch_file in patch_files:
        #print(patch_file)
        patch_path = os.path.join(input_directory, patch_file)
        patch = cv2.imread(patch_path)

        y_start = int(patch_file.split('_')[-2])
        x_start = int((patch_file.split('_')[-1]).split('.')[0])

        # Calcular las coordenadas finales del parche
        y_end, x_end = y_start + patch_size, x_start + patch_size

        # Actualizar los pesos en la zona de solapamiento
        reconstructed_image[y_start:y_end, x_start:x_end] += patch * weight_mask[:, :, np.newaxis]
        weights[y_start:y_end, x_start:x_end] += weight_mask[:, :, np.newaxis]

    # Asegurar que no haya división por cero
    weights[weights == 0] = 1.0

    # Promediar los valores ponderados para obtener la imagen reconstruida
    reconstructed_image /= weights
    reconstructed_image = reconstructed_image.astype(np.uint8)

    # Recortar la imagen al tamaño original
    reconstructed_image = reconstructed_image[:original_size[0], :original_size[1]]
    reconstructed_filepath = os.path.join(output_filepath, base_image_name+".jpg")
    #print(reconstructed_filepath)

    if GrayScale:
        gray_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(reconstructed_filepath, gray_image)
    else:
        cv2.imwrite(reconstructed_filepath, reconstructed_image)


def reconstruct_all_images_from_patches_overlap(input_directory_patches, output_directory_reconstructed, patch_size=512, original_size=(1080, 1920, 3), selected_frames=[], GrayScale=False):

    hamming_window = np.hamming(patch_size)
    xx, yy = np.meshgrid(hamming_window, hamming_window)
    weight_mask = xx * yy

    os.makedirs(output_directory_reconstructed, exist_ok=True)
    # List all patch files in the input directoSFM__
    patch_files = [f for f in os.listdir(input_directory_patches) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Extract unique basenames from patch files
    unique_basenames = [os.path.splitext(f.split('_patch')[0])[0] for f in patch_files]
    if selected_frames:
        selected_frames = set([os.path.splitext(f.split('.jpg')[0])[0] for f in selected_frames])
        unique_basenames = list(set(unique_basenames) & set(selected_frames))

    unique_basenames = sorted(unique_basenames)
    # Iterate over each unique basename
    for base_image_name in unique_basenames:
        try:
        # Reconstruct image from patches for the current basename
            reconstruct_image_from_patches_overlap(base_image_name, input_directory_patches, output_directory_reconstructed, weight_mask, patch_size, original_size, GrayScale)
        except Exception as e:
            print(f"Error processing frame {base_image_name}: {str(e)}")


def binarize_flares_percentiles(input_directory, output_directory, selected_frames=[], new_size=[], percentile=80, max_level=0):
    # Ensure output irectory exists
    os.makedirs(output_directory, exist_ok=True)

    # List all image files in the input directory
    image_files = [f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if selected_frames:
        image_files = list(set(image_files) & set(selected_frames))


    # Iterate over each image file
    for image_file in image_files:
        # Read the grayscale image
        image_path = os.path.join(input_directory, image_file)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Calculate the specified percentile value
        threshold_value = np.percentile(gray_image, percentile)

        M, N = gray_image.shape

        if max_level < threshold_value:
            # Threshold the image and create a new image
            thresholded_image = np.where(gray_image > threshold_value, 0, 255)
        else:
            thresholded_image = np.full((M, N), 255, dtype=np.uint8)

        if new_size:
            P, Q = new_size
            # Ajustar el número de filas (M)
            if M < P:
                thresholded_image = np.vstack((thresholded_image, np.full((P - M, N), 255)))
            elif M > P:
                thresholded_image = thresholded_image[:P, :]

            # Ajustar el número de columnas (N)
            if N < Q:
                thresholded_image = np.hstack((thresholded_image, np.full((M, Q - N), 255)))
            elif N > Q:
                thresholded_image = thresholded_image[:, :Q]

        # Save the thresholded image in the output directory
        output_filename = os.path.splitext(image_file)[0] + ".png"
        output_path = os.path.join(output_directory, output_filename)
        cv2.imwrite(output_path, thresholded_image)

def autocontrast_func(img, cutoff=0):
    '''
        same output as PIL.ImageOps.autocontrast
    '''
    n_bins = 256
    def tune_channel(ch):
        n = ch.size
        cut = cutoff * n // 100
        if cut == 0:
            high, low = ch.max(), ch.min()
        else:
            hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
            low = np.argwhere(np.cumsum(hist) > cut)
            low = 0 if low.shape[0] == 0 else low[0]
            high = np.argwhere(np.cumsum(hist[::-1]) > cut)
            high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]
        if high <= low:
            table = np.arange(n_bins)
        else:
            scale = (n_bins - 1) / (high - low)
            offset = -low * scale
            table = np.arange(n_bins) * scale + offset
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.clip(0, 255).astype(np.uint8)
        return table[ch]
    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def binarize_flares_level(input_directory, output_directory, level=128):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # List all image files in the input directory
    image_files = [f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Iterate over each image file
    for image_file in image_files:
        # Read the grayscale image
        image_path = os.path.join(input_directory, image_file)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Threshold the image and create a new image
        thresholded_image = np.where(gray_image > level, 0, 255)

        # Save the thresholded image in the output directory
        output_filename = os.path.splitext(image_file)[0] + ".png"
        output_path = os.path.join(output_directory, output_filename)
        cv2.imwrite(output_path, thresholded_image)


def get_masks(input_mask_dataset, image_dir, output_mask_dir):
    files = os.listdir(image_dir)
    files = [os.path.splitext(file)[0] for file in files]

    masks_in_dataset = os.listdir(input_mask_dataset)
    masks_in_dataset= [os.path.splitext(file)[0] for file in masks_in_dataset]

    os.makedirs(output_mask_dir, exist_ok=True)

    files_to_copy = set(files) & set(masks_in_dataset)


    for filename in files_to_copy:
        source_file = os.path.join(input_mask_dataset, filename +".png")
        destination_file = os.path.join(output_mask_dir, filename + ".png")
        shutil.copy(source_file, destination_file)






"""
El enviroment es tf y el proyecto flareRemoval Original
Como se entrena el modelo: python3 -m python.train --train_dir=./logsJPG2debug --scene_dir=../../TransmisionLayer_Candidates/new_scenes --flare_dir=../../TransmisionLayer_Candidates/flares
Como se procesan los flares.....
"""
"""
input_folder = "/media/virginia/HOME_grande/virginia/QuarriesDATA/Valdilecha_24_01_2024/Oppo_Reno/OnlyWithSRT/VID_20240124_100041Z/images"
output_folder_patches = "/media/virginia/HOME_grande/virginia/QuarriesDATA/Valdilecha_24_01_2024/Oppo_Reno/OnlyWithSRT/VID_20240124_100041Z/imagesPatches"
#output_folder_reconstructed = "/media/virginia/HOME_grande/virginia/QuarriesDATA/Valdilecha_24_01_2024/Oppo_Reno/OnlyWithSRT/VID_20240124_092540Z/imagesReconstructed"
patchesFlares ="/media/virginia/HOME_grande/virginia/QuarriesDATA/Valdilecha_24_01_2024/Oppo_Reno/OnlyWithSRT/VID_20240124_100041Z/imagesPatchesProc/output_flare"
reconstructedFlares = "/media/virginia/HOME_grande/virginia/QuarriesDATA/Valdilecha_24_01_2024/Oppo_Reno/OnlyWithSRT/VID_20240124_100041Z/Flares"

#patchesRecovery ="/media/virginia/HOME_grande/virginia/QuarriesDATA/Valdilecha_24_01_2024/Oppo_Reno/OnlyWithSRT/VID_20240124_092540Z/imagesPatchesProc/output"
#reconstructedImagProc = "/media/virginia/HOME_grande/virginia/QuarriesDATA/Valdilecha_24_01_2024/Oppo_Reno/OnlyWithSRT/VID_20240124_092540Z/imageRecovery"
binary_masks = "/media/virginia/HOME_grande/virginia/QuarriesDATA/Valdilecha_24_01_2024/Oppo_Reno/OnlyWithSRT/VID_20240124_100041Z/masks"

reconstruct_all_images_from_patches(patchesFlares, reconstructedFlares, GrayScale=True)
binarize_flares_percentiles(reconstructedFlares, binary_masks, percentile=90)
input_directory_patches = "/media/virginia/HOME_grande/virginia/QuarriesDATA/Valdilecha_24_01_2024/Oppo_Reno/OnlyWithSRT/VID_20240124_100041Z/imagesPatchesProcOverlap/output"
output_directory_reconstructed = "/media/virginia/HOME_grande/virginia/QuarriesDATA/Valdilecha_24_01_2024/Oppo_Reno/OnlyWithSRT/VID_20240124_100041Z/ImagesFromFlareRemoval"
reconstruct_all_images_from_patches_overlap(input_directory_patches, output_directory_reconstructed, patch_size=512,
                                            original_size=(1080, 1920, 3), GrayScale=False)
reconstruct_all_images_from_patches(patchesRecovery, reconstructedImagProc)
"""
