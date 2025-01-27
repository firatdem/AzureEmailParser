import fitz
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
from tkinter import filedialog
import cv2
import numpy as np
import json
import os
from classes import DetectionResult, NpEncoder
import concurrent.futures
import re


def extract_text_and_images(pdf_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    images = []

    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # Convert PDF pages to images using pdf2image
        print(f"Extracting text from page {page_num + 1}...")
        images.append(convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)[0])

        # Save the extracted image to the output folder
        image_filename = os.path.join(output_folder, f'page_{page_num + 1}.png')
        images[-1].save(image_filename, 'PNG')

        print(f"Text extracted from page {page_num + 1}. Image saved as {image_filename}.")

    return images


# THIS FUNCTION CUTS OFF THE RIGHT SIDE 800 PX TO IGNORE TEXT FALSE POSITIVES
# WORKS
def extract_text_and_imagesCUTOFF(pdf_path, output_folder):
    """
    Extract text and images from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Folder where extracted images will be saved.

    Returns:
        images (list of PIL.Image.Image): List of extracted images.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    images = []

    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # Convert PDF pages to images using pdf2image
        print(f"Extracting text from page {page_num + 1}...")
        pdf2image_images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)

        if pdf2image_images is not None:
            # Cover the first 800 pixels from the right with white
            for pdf2image_image in pdf2image_images:
                width, height = pdf2image_image.size
                white_rect = Image.new('RGB', (800, height), (255, 255, 255))
                pdf2image_image.paste(white_rect, (width - 800, 0))
                images.append(pdf2image_image)

            # Save the extracted image to the output folder
            image_filename = os.path.join(output_folder, f'page_{page_num + 1}.png')
            pdf2image_images[0].save(image_filename, 'PNG')

            print(f"Text extracted from page {page_num + 1}. Image saved as {image_filename}.")

    return images


def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    print("File path selected by open_file_dialog:", file_path)
    return file_path


def search_for_png_image(image_data, custom_image_paths, threshold=0.5, distance_threshold=100, target_size=(70, 70)):
    """
    Search for a custom PNG image within extracted image data using multiple reference images.
    :param image_data: Extracted image data (a list of images).
    :param custom_image_paths: List of paths to custom PNG images (reference images).
    :param threshold: Template matching threshold (default is 0.5).
    :param distance_threshold: Minimum distance between detected spots to consider them unique.
    :param target_size: Target size for resizing custom images.
    :return: A tuple containing the count of unique occurrences and a list of coordinates where the custom PNG images were found.
    """

    # Resize the custom PNG images to the target size
    resized_custom_images = [cv2.resize(cv2.imread(custom_image_path, cv2.IMREAD_GRAYSCALE), target_size) for custom_image_path in custom_image_paths]

    # Load the first extracted image to initialize the result image
    result_image = np.array(image_data[0])

    # Initialize a set to store unique locations for the same symbol across all reference images
    unique_locations = set()

    for custom_image in resized_custom_images:
        for i, extracted_image in enumerate(image_data):
            # Load and process the extracted image
            np_image = np.array(extracted_image)
            gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

            # Template matching
            result = cv2.matchTemplate(gray_image, custom_image, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)

            # Check and add unique locations to the set
            for loc in zip(*locations[::-1]):
                is_unique = all(np.linalg.norm(np.array(loc) - np.array(existing_loc)) > distance_threshold
                               for existing_loc in unique_locations)
                if is_unique:
                    unique_locations.add(loc)

                    # Mark the detected region on the result image
                    (x, y) = loc
                    w, h = custom_image.shape[::-1]  # Get the width and height of the custom image
                    # Correct the rectangle coordinates
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save the result image with all symbols marked
    cv2.imwrite('marked_image_first_search.png', result_image)

    # Count the unique occurrences for the same symbol across all reference images
    total_count = len(unique_locations)

    # Save the unique locations to a file
    with open('unique_locations.json', 'w') as file:
        json.dump([list(loc) for loc in unique_locations], file, default=lambda x: x.__str__())

    return total_count, result_image, list(unique_locations)


def populate_class_list_from_coordinates(coordinates_json):
    print("Content of 'coordinates_json':", repr(coordinates_json))

    try:
        coordinates_list = json.loads(coordinates_json)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON coordinates: {e}")
        return []
    detection_results = []
    for i, entry in enumerate(coordinates_list, start=1):
        # Extract coordinate values from the 'coordinates' key
        x, y = map(int, entry['coordinates'])

        # Create a DetectionResult instance
        detection_result = DetectionResult(result_number=i, coordinate_location=(x, y), flag=1)
        detection_results.append(detection_result)

    return detection_results


#Saves the results to the json, this is for seamless interaction between functions
def search_for_png_class_add(image_data, custom_image_paths, threshold=0.4, distance_threshold=70, target_size=(50, 50)):
    # Resize the custom PNG images to the target size
    resized_custom_images = [cv2.resize(cv2.imread(custom_image_path, cv2.IMREAD_GRAYSCALE), target_size) for custom_image_path in custom_image_paths]

    # Load the first extracted image to initialize the result image
    result_image = np.array(image_data[0])

    # Initialize a list to store DetectionResult instances
    detection_results = []

    # Initialize a set to store unique locations for the same symbol across all reference images
    unique_locations = set()

    for custom_image in resized_custom_images:
        for i, extracted_image in enumerate(image_data, start=1):
            # Load and process the extracted image
            np_image = np.array(extracted_image)
            gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

            # Template matching
            result = cv2.matchTemplate(gray_image, custom_image, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)

            # Check and add unique locations to the list
            for loc in zip(*locations[::-1]):
                is_unique = all(np.linalg.norm(np.array(loc) - np.array(existing_result.coordinate_location)) > distance_threshold
                                for existing_result in detection_results)
                if is_unique:
                    # Update the set with the unique location
                    unique_locations.add(loc)

                    # Create a DetectionResult instance and add it to the list
                    detection_result = DetectionResult(coordinate_location=loc, flag=0)
                    detection_results.append(detection_result)

                    # Mark the detected region on the result image
                    (x, y) = loc
                    w, h = custom_image.shape[::-1]  # Get the width and height of the custom image
                    # Correct the rectangle coordinates
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save the result image with all symbols marked
    cv2.imwrite('marked_image_first_search.png', result_image)

    # Save the unique locations to a file using the custom encoder
    coordinates_json_path = 'unique_locations.json'
    with open(coordinates_json_path, 'w') as file:
        json.dump([
            {"result_id": i, "coordinates": list(map(int, loc)), "flag": int(0)}
            for i, loc in enumerate(unique_locations, start=1)
        ], file, cls=NpEncoder)

    return len(detection_results), result_image, coordinates_json_path



# THIS WORKS, DONT USE 2
def save_images_by_coordinates(images, coordinates_file, output_folder,
                               target_size=(70, 70), search_area_scale=2.0):
    """
    Save images based on coordinates in a JSON file.
    :param images: Extracted image data (a list of images).
    :param coordinates_file: Path to the JSON file containing coordinates.
    :param output_folder: Directory to save the location-based images.
    """
    # Load coordinates from the JSON file
    with open(coordinates_file, 'r') as file:
        coordinates_data = json.load(file)

    # Extract only the 'coordinates' field from the JSON data
    valid_coordinates = [entry.get("coordinates") for entry in coordinates_data]

    # Remove None values from the list
    valid_coordinates = [coord for coord in valid_coordinates if coord is not None]

    # Initialize the result image
    result_image = np.array(images[0])

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for coord in valid_coordinates:
        # Check if coord is a valid coordinate list
        if isinstance(coord, list) and len(coord) == 2:
            # Extract coordinates and calculate the search area
            x, y = map(float, coord)
            search_area = (
                max(0, int(x - (search_area_scale - 1) * target_size[0] / 2)),
                max(0, int(y - (search_area_scale - 1) * target_size[1] / 2)),
                min(result_image.shape[1], int(x + search_area_scale * target_size[0])),
                min(result_image.shape[0], int(y + search_area_scale * target_size[1]))
            )

            # Extract the search area from the image and convert to grayscale
            search_area_image = cv2.cvtColor(np.array(images[0])[search_area[1]:search_area[3], search_area[0]:search_area[2]], cv2.COLOR_RGB2GRAY)

            # Save the location-based image
            image_name = f"location_{int(x)}_{int(y)}.png"
            image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(image_path, search_area_image)

            # Mark the detected region on the result image
            cv2.rectangle(result_image, (search_area[0], search_area[1]),
                          (search_area[2], search_area[3]), (0, 255, 0), 2)

    # Save the result image with all symbols marked
    cv2.imwrite(os.path.join(output_folder, 'marked_image.png'), result_image)

    # Print debug information
    print("Number of saved images:", len(valid_coordinates))
    print("Output directory:", output_folder)

    return len(valid_coordinates), result_image, valid_coordinates


def mark_up_images(images, coordinates_json_path, output_folder='marked_images'):
    # Read the JSON content from the file
    try:
        with open(coordinates_json_path, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON coordinates: {e}")
        return 0, []

    # Create an output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    unique_locations_list = []

    for idx, image in enumerate(images):
        # Check if the image is a PIL Image
        if isinstance(image, Image.Image):
            # Convert PIL Image to NumPy array
            image = np.array(image)

        # Check if the image is a valid NumPy array
        if not isinstance(image, np.ndarray):
            print(f"Error: Image {idx + 1} is not a valid NumPy array. Type: {type(image)}")
            continue

        # Initialize a copy of the original image for marking
        marked_image = np.copy(image)

        # Initialize a list to store DetectionResult instances
        detection_results = []

        # Initialize a set to store unique locations for the same symbol across all reference images
        unique_locations = set()

        # Iterate through coordinates and mark up the image
        for entry in data:
            result_id = entry.get("result_id")
            coordinates = entry.get("coordinates")
            flag = entry.get("flag")

            # Check if the 'coordinates' key is present and flag is 1
            if coordinates is not None and flag == 1:
                # Extract x and y from coordinates
                x, y = map(int, coordinates[:2])

                # Check if the marked location is within the image bounds
                if 0 <= x < marked_image.shape[1] and 0 <= y < marked_image.shape[0]:
                    # Check if the location is unique
                    is_unique = all(
                        np.linalg.norm(np.array((x, y)) - np.array(existing_result.coordinate_location)) > 10
                        for existing_result in detection_results)

                    if is_unique:
                        # Update the set with the unique location
                        unique_locations.add((x, y))

                        # Create a DetectionResult instance and add it to the list
                        detection_result = DetectionResult(coordinate_location=(x, y), flag=0)
                        detection_results.append(detection_result)

                        # Mark the detected region on the marked image
                        cv2.circle(marked_image, (x, y), radius=100, color=(0, 0, 255), thickness=2)

        # Save the marked-up image only if there are unique locations
        if unique_locations:
            output_path = os.path.join(output_folder, f"marked_image_{idx + 1}.png")
            cv2.imwrite(output_path, marked_image)

            print(f"Image {idx + 1} marked and saved at: {output_path}")

            # Add the unique locations for the current image to the list
            unique_locations_list.append(list(unique_locations))

    return len(unique_locations_list), unique_locations_list


def read_coordinates_from_json(json_file_path):
    """
    Read coordinates from a JSON file and return a list of coordinate tuples.
    :param json_file_path: Path to the JSON file.
    :return: List of coordinate tuples.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    return data


def save_images_by_coordinates_to_npy(image_data, coordinates_file, output_npy,
                                       target_size=(70, 70), search_area_scale=2.0):
    """
    Save images based on coordinates in a JSON file to a NumPy dataset.
    :param image_data: Extracted image data (a list of images).
    :param coordinates_file: Path to the JSON file containing coordinates.
    :param output_npy: Path to save the NumPy dataset.
    :param target_size: Target size for resizing custom images.
    :param search_area_scale: Scaling factor for the search area around the coordinates (default is 1.2).
    :return: Number of saved images, result image, coordinates
    """

    # Load coordinates from the JSON file
    with open(coordinates_file, 'r') as file:
        coordinates = json.load(file)

    # Initialize the result image
    result_image = np.array(image_data[0])

    # Initialize lists to store images and corresponding coordinates
    saved_images = []
    saved_coordinates = []

    for coord in coordinates:
        # Extract coordinates and calculate the search area
        (x, y) = map(float, coord)
        search_area = (
            max(0, int(x - (search_area_scale - 1) * target_size[0] / 2)),
            max(0, int(y - (search_area_scale - 1) * target_size[1] / 2)),
            min(result_image.shape[1], int(x + search_area_scale * target_size[0])),
            min(result_image.shape[0], int(y + search_area_scale * target_size[1]))
        )

        # Extract the search area from the image and convert to grayscale
        search_area_image = cv2.cvtColor(np.array(image_data[0])[search_area[1]:search_area[3], search_area[0]:search_area[2]], cv2.COLOR_RGB2GRAY)

        # Save the location-based image
        saved_images.append(search_area_image)
        saved_coordinates.append((x, y))

        # Mark the detected region on the result image
        cv2.rectangle(result_image, (search_area[0], search_area[1]),
                      (search_area[2], search_area[3]), (0, 255, 0), 2)

    # Save the result image with all symbols marked
    cv2.imwrite(os.path.join(output_npy, 'marked_image.png'), result_image)

    # Convert lists to NumPy arrays and save as a dataset
    saved_images = np.array(saved_images)
    saved_coordinates = np.array(saved_coordinates)
    np.savez(output_npy, images=saved_images, coordinates=saved_coordinates)

    # Print debug information
    print("Number of saved images:", len(coordinates))
    print("Output directory:", output_npy)

    return len(coordinates), result_image, coordinates


def save_images_by_coordinates_2(image_data, coordinates_file, output_dir,
                                target_size=(70, 70), search_area_scale=1.0):
    """
    Save images based on coordinates in a JSON file.
    :param image_data: Extracted image data (a list of images).
    :param coordinates_file: Path to the JSON file containing coordinates.
    :param output_dir: Directory to save the location-based images.
    :param target_size: Target size for resizing custom images.
    :param search_area_scale: Scaling factor for the search area around the coordinates (default is 2.0).
    """

    # Load coordinates from the JSON file
    with open(coordinates_file, 'r') as file:
        coordinates_data = json.load(file)

    # Extract only the 'coordinate_location' field from the JSON data
    valid_coordinates = [entry.get("coordinate_location") for entry in coordinates_data]

    # Remove None values from the list
    valid_coordinates = [coord for coord in valid_coordinates if coord is not None]

    # Initialize the result image
    result_image = np.array(image_data[0])

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for coord in valid_coordinates:
        # Extract coordinates and calculate the search area
        (x, y) = map(float, coord)
        search_area = (
            max(0, int(x - search_area_scale * target_size[0] / 2)),
            max(0, int(y - search_area_scale * target_size[1] / 2)),
            min(result_image.shape[1], int(x + search_area_scale * target_size[0] / 2)),
            min(result_image.shape[0], int(y + search_area_scale * target_size[1] / 2))
        )

        # Extract the search area from the image and convert to grayscale
        search_area_image = np.array(image_data[0])[search_area[1]:search_area[3], search_area[0]:search_area[2]]

        # Save the location-based image
        image_name = f"location_{int(x)}_{int(y)}.png"
        image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(image_path, cv2.cvtColor(search_area_image, cv2.COLOR_RGB2GRAY))

        # Mark the detected region on the result image
        cv2.rectangle(result_image, (search_area[0], search_area[1]),
                      (search_area[2], search_area[3]), (0, 255, 0), 2)

    # Save the result image with all symbols marked
    cv2.imwrite(os.path.join(output_dir, 'marked_image.png'), result_image)

    # Print debug information
    print("Number of saved images:", len(valid_coordinates))
    print("Output directory:", output_dir)

    return len(valid_coordinates), result_image, valid_coordinates


# THIS PROGRAM USES THE SAME LOGIC AS SAVE BY COORDINATE BUT SAVES THE WHOLE GRID
# IT SAVES THEM AS OVERLAPPING SQUARES AS NOT TO MISS ANYTHING
def save_images_by_grid(images, output_folder, target_size=(70, 70), overlap_ratio=0.5):
    """
    Save images based on a grid of overlapping squares.
    :param images: Extracted image data (a list of images).
    :param output_folder: Directory to save the grid-based images.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize the result image
    result_image = np.array(images[0])

    # Get the dimensions of the result image
    result_height, result_width, _ = result_image.shape

    # Calculate the step size for creating the grid
    step_size_x = int(target_size[0] * (1 - overlap_ratio))
    step_size_y = int(target_size[1] * (1 - overlap_ratio))

    for y in range(0, result_height - target_size[1] + 1, step_size_y):
        for x in range(0, result_width - target_size[0] + 1, step_size_x):
            # Extract the search area from the image and convert to grayscale
            search_area_image = cv2.cvtColor(
                result_image[y:y + target_size[1], x:x + target_size[0]], cv2.COLOR_RGB2GRAY
            )

            # Save the grid-based image
            image_name = f"grid_{x}_{y}.png"
            image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(image_path, search_area_image)

            # Mark the detected region on the result image
            cv2.rectangle(result_image, (x, y), (x + target_size[0], y + target_size[1]), (0, 255, 0), 2)

    # Save the result image with all squares marked
    cv2.imwrite(os.path.join(output_folder, 'marked_image.png'), result_image)

    # Print debug information
    num_saved_images = (result_height // step_size_y) * (result_width // step_size_x)
    print("Number of saved images:", num_saved_images)
    print("Output directory:", output_folder)

    return num_saved_images, result_image


def whole_image_png_class_add(image_data, coord, target_size=(500, 500), overlap_factor=0.5):

    # Load the first extracted image to initialize the result image
    result_image = np.array(image_data[0])

    # Initialize a list to store DetectionResult instances
    detection_results = []

    # Initialize a list to store all locations
    all_locations = []

    # Iterate over all extracted images
    for extracted_image in image_data:
        np_image = np.array(extracted_image)

        # Get the dimensions of the image
        image_height, image_width = np_image.shape[:2]

        # Iterate over all possible squares with overlap
        for x in range(0, image_width - target_size[1], int(target_size[1] * (1 - overlap_factor))):
            for y in range(0, image_height - target_size[0], int(target_size[0] * (1 - overlap_factor))):
                # Extract the square from the image
                square = np_image[y:y+target_size[0], x:x+target_size[1]]

                # Store the location of the square
                all_locations.append((x, y))

                # Create a DetectionResult instance for the square with flag set to 0
                detection_result = DetectionResult(coordinate_location=(x, y), flag=0)
                detection_results.append(detection_result)

                # Mark the detected region on the result image
                cv2.rectangle(result_image, (x, y), (x + target_size[1], y + target_size[0]), (0, 0, 255), 2)

    # Save the result image with all symbols marked
    cv2.imwrite('marked_image_first_search.png', result_image)

    # Save the locations to a file using the custom encoder
    coordinates_json_path = 'unique_locations.json'
    with open(coordinates_json_path, 'w') as file:
        json.dump([
            {"result_id": i, "coordinates": list(map(int, loc)), "flag": int(0)}  # Set flag to 0 for all detections
            for i, loc in enumerate(all_locations, start=1)
        ], file, cls=NpEncoder)

    return len(detection_results), result_image, coordinates_json_path


def whole_image_png_class_add_2(image_data, coord, target_size=(500, 500), overlap_factor=0.6):

    # Load the first extracted image to initialize the result image
    result_image = np.array(image_data[0])

    # Initialize a list to store DetectionResult instances
    detection_results = []

    # Initialize a list to store all locations
    all_locations = []

    # Iterate over all extracted images
    for extracted_image in image_data:
        np_image = np.array(extracted_image)

        # Get the dimensions of the image
        image_height, image_width = np_image.shape[:2]

        # Iterate over all possible squares without overlap
        for x in range(0, image_width - target_size[1], target_size[1]):
            for y in range(0, image_height - target_size[0], target_size[0]):
                # Extract the square from the image
                square = np_image[y:y + target_size[0], x:x + target_size[1]]

                # Store the location of the square
                all_locations.append((x, y))

                # Create a DetectionResult instance for the square with flag set to 0
                detection_result = DetectionResult(coordinate_location=(x, y), flag=0)
                detection_results.append(detection_result)

                # Mark the detected region on the result image
                cv2.rectangle(result_image, (x, y), (x + target_size[1], y + target_size[0]), (0, 0, 255), 2)

        # Iterate over all possible squares with overlap
        for x in range(0, image_width - target_size[1], int(target_size[1] * (1 - overlap_factor))):
            for y in range(0, image_height - target_size[0], int(target_size[0] * (1 - overlap_factor))):
                # Extract the square from the image
                square = np_image[y:y + target_size[0], x:x + target_size[1]]

                # Store the location of the square
                all_locations.append((x, y))

                # Create a DetectionResult instance for the square with flag set to 0
                detection_result = DetectionResult(coordinate_location=(x, y), flag=0)
                detection_results.append(detection_result)

                # Mark the detected region on the result image
                cv2.rectangle(result_image, (x, y), (x + target_size[1], y + target_size[0]), (0, 0, 255), 2)

    # Save the result image with all symbols marked
    cv2.imwrite('marked_image_first_search.png', result_image)

    # Save the locations to a file using the custom encoder
    coordinates_json_path = 'unique_locations.json'
    with open(coordinates_json_path, 'w') as file:
        json.dump([
            {"result_id": i, "coordinates": list(map(int, loc)), "flag": int(0)}  # Set flag to 0 for all detections
            for i, loc in enumerate(all_locations, start=1)
        ], file, cls=NpEncoder)

    return len(detection_results), result_image, coordinates_json_path


def mark_images_by_coordinates(images, coordinates_file, search_area_scale=2.0):
    # Load coordinates from the JSON file
    try:
        with open(coordinates_file, 'r') as file:
            coordinates_data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON coordinates: {e}")
        return 0, np.array(images[0]), []

    # Extract only the 'coordinates' and 'flag' fields from the JSON data
    valid_coordinates = [(entry.get("coordinates"), entry.get("flag")) for entry in coordinates_data]

    # Remove None values and filter based on the flag value
    valid_coordinates = [(coord, flag) for coord, flag in valid_coordinates if coord is not None and flag == 1]

    # Initialize the result image
    result_image = np.array(images[0])

    for coord, _ in valid_coordinates:
        # Check if coord is a valid coordinate list
        if isinstance(coord, list) and len(coord) == 2:
            # Extract coordinates and calculate the search area
            x, y = map(float, coord)
            target_size = (70, 70)  # Assuming fixed target size
            search_area = (
                max(0, int(x - (search_area_scale - 1) * target_size[0] / 2)),
                max(0, int(y - (search_area_scale - 1) * target_size[1] / 2)),
                min(result_image.shape[1], int(x + search_area_scale * target_size[0])),
                min(result_image.shape[0], int(y + search_area_scale * target_size[1]))
            )

            # Mark the detected region on the result image
            cv2.rectangle(result_image, (search_area[0], search_area[1]),
                          (search_area[2], search_area[3]), (0, 255, 0), 2)

    print("Number of valid coordinates:", len(valid_coordinates))
    print("Valid coordinates:", valid_coordinates)

    return len(valid_coordinates), result_image, valid_coordinates


def save_images_by_coordinates_3(images, coordinates_file, output_folder,
                               target_size=(70, 70), search_area_scale=2.0):
    """
    Save images based on coordinates in a JSON file.
    :param images: Extracted image data (a list of images).
    :param coordinates_file: Path to the JSON file containing coordinates.
    :param output_folder: Directory to save the location-based images.
    """
    # Load coordinates from the JSON file
    with open(coordinates_file, 'r') as file:
        coordinates_data = json.load(file)

    # Extract only the 'coordinates' field from the JSON data
    valid_coordinates = [entry.get("coordinates") for entry in coordinates_data]

    # Remove None values from the list
    valid_coordinates = [coord for coord in valid_coordinates if coord is not None]

    # Initialize the result image
    result_image = np.array(images[0])

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for i, coord in enumerate(valid_coordinates):
        # Check if coord is a valid coordinate list
        if isinstance(coord, list) and len(coord) == 2:
            # Extract coordinates and calculate the search area
            x, y = map(float, coord)
            search_area = (
                max(0, int(x - (search_area_scale - 1) * target_size[0] / 2)),
                max(0, int(y - (search_area_scale - 1) * target_size[1] / 2)),
                min(result_image.shape[1], int(x + search_area_scale * target_size[0])),
                min(result_image.shape[0], int(y + search_area_scale * target_size[1]))
            )

            # Extract the search area from the image and convert to grayscale
            search_area_image = cv2.cvtColor(np.array(images[0])[search_area[1]:search_area[3], search_area[0]:search_area[2]], cv2.COLOR_RGB2GRAY)

            # Get the result_id from the coordinates_data
            result_id = coordinates_data[i].get("result_id")

            # Save the location-based image with result_id in the name
            image_name = f"location_{result_id}_{int(x)}_{int(y)}.png"
            image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(image_path, search_area_image)

            # Mark the detected region on the result image
            cv2.rectangle(result_image, (search_area[0], search_area[1]),
                          (search_area[2], search_area[3]), (0, 255, 0), 2)

    # Save the result image with all symbols marked
    cv2.imwrite(os.path.join(output_folder, 'marked_image.png'), result_image)

    # Print debug information
    print("Number of saved images:", len(valid_coordinates))
    print("Output directory:", output_folder)

    return len(valid_coordinates), result_image, valid_coordinates


def template_matching_worker(args):
    custom_image, np_image, threshold, distance_threshold, detection_results = args
    gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

    result = cv2.matchTemplate(gray_image, custom_image, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    for loc in zip(*locations[::-1]):
        is_unique = all(np.linalg.norm(np.array(loc) - np.array(existing_result.coordinate_location)) > distance_threshold
                        for existing_result in detection_results)
        if is_unique:
            detection_result = DetectionResult(coordinate_location=loc, flag=0)
            detection_results.append(detection_result)

    return detection_results


def search_for_png_class_add_2(image_data, custom_image_paths, threshold=0.4, distance_threshold=50, target_size=(50, 50)):
    resized_custom_images = [cv2.resize(cv2.imread(custom_image_path, cv2.IMREAD_GRAYSCALE), target_size) for custom_image_path in custom_image_paths]
    result_image = np.array(image_data[0])
    detection_results = []
    unique_locations = set()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        args_list = [(custom_image, np.array(extracted_image), threshold, distance_threshold, detection_results) for custom_image in resized_custom_images for extracted_image in image_data]
        results = executor.map(template_matching_worker, args_list)

        for result in results:
            detection_results.extend(result)

    # Update the unique_locations set using the coordinate_location attribute of DetectionResult instances
    unique_locations.update(result.coordinate_location for result in detection_results)

    # Save the result image with all symbols marked
    cv2.imwrite('marked_image_first_search.png', result_image)

    coordinates_json_path = 'unique_locations.json'
    with open(coordinates_json_path, 'w') as file:
        json.dump([
            {"result_id": i, "coordinates": list(map(int, loc.coordinate_location)), "flag": int(0)}
            for i, loc in enumerate(detection_results, start=1)
        ], file, cls=NpEncoder)

    return len(detection_results), result_image, coordinates_json_path


# THIS IS SAME, BUT IS INTENDED TO NOT BE HARD CODED TO MARK FLAGS 1
# IT TAKES TO BE MARKED FLAG AS ARGUMENT
def mark_up_images_2(images, coordinates_json_path, output_folder='marked_images'):
    # Read the JSON content from the file
    try:
        with open(coordinates_json_path, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON coordinates: {e}")
        return 0, []

    # Create an output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    unique_locations_list = []

    for idx, image in enumerate(images):
        # Check if the image is a PIL Image
        if isinstance(image, Image.Image):
            # Convert PIL Image to NumPy array
            image = np.array(image)

        # Check if the image is a valid NumPy array
        if not isinstance(image, np.ndarray):
            print(f"Error: Image {idx + 1} is not a valid NumPy array. Type: {type(image)}")
            continue

        # Initialize a copy of the original image for marking
        marked_image = np.copy(image)

        # Initialize lists to store DetectionResult instances for flag values 1 and 2
        detection_results_flag_1 = []
        detection_results_flag_2 = []

        # Initialize sets to store unique locations for the same symbol across all reference images for flag values 1 and 2
        unique_locations_flag_1 = set()
        unique_locations_flag_2 = set()

        # Iterate through coordinates and mark up the image
        for entry in data:
            result_id = entry.get("result_id")
            coordinates = entry.get("coordinates")
            flag = entry.get("flag")

            # Check if the 'coordinates' key is present
            if coordinates is not None:
                # Extract x and y from coordinates
                x, y = map(int, coordinates[:2])

                # Check if the marked location is within the image bounds
                if 0 <= x < marked_image.shape[1] and 0 <= y < marked_image.shape[0]:
                    # Check if the location is unique
                    is_unique = all(
                        np.linalg.norm(np.array((x, y)) - np.array(existing_result.coordinate_location)) > 10
                        for existing_result in detection_results_flag_1 if flag == 1
                    ) and all(
                        np.linalg.norm(np.array((x, y)) - np.array(existing_result.coordinate_location)) > 10
                        for existing_result in detection_results_flag_2 if flag == 2
                    )

                    # Update the set with the unique location
                    unique_locations = unique_locations_flag_1 if flag == 1 else unique_locations_flag_2

                    if is_unique:
                        # Update the set with the unique location
                        unique_locations.add((x, y))

                        # Create a DetectionResult instance and add it to the list based on the flag value
                        detection_result = DetectionResult(coordinate_location=(x, y), flag=0)
                        detection_results = detection_results_flag_1 if flag == 1 else detection_results_flag_2
                        detection_results.append(detection_result)

                        # Mark the detected region on the marked image based on the flag value
                        if flag == 1:
                            color = (0, 0, 255)  # Red for flag 1
                            cv2.circle(marked_image, (x, y), radius=100, color=color, thickness=2)
                        elif flag == 2:
                            color = (0, 255, 0)  # Green for flag 2
                            cv2.circle(marked_image, (x, y), radius=100, color=color, thickness=2)

        # Save the marked-up image only if there are unique locations
        if unique_locations_flag_1 or unique_locations_flag_2:
            output_path = os.path.join(output_folder, f"marked_image_{idx + 1}.png")
            cv2.imwrite(output_path, marked_image)

            print(f"Image {idx + 1} marked and saved at: {output_path}")

            # Add the unique locations for the current image to the list
            unique_locations_list.extend([list(unique_locations_flag_1), list(unique_locations_flag_2)])

    return len(unique_locations_list), unique_locations_list

def parse_extracted_text(extracted_text):
    """
    Parse the extracted text to find specific data.

    Args:
        extracted_text (str): The OCR-extracted text.

    Returns:
        dict: A dictionary containing the extracted data.
    """
    data = {
        "Date": "",
        "Purchase Order": "",
        "Company": "",
        "Location": "",
        "Amount": "",
        "Description": "",
        "Comments": ""
    }

    lines = extracted_text.splitlines()
    num_lines = len(lines)
    idx = 0

    # Flags to indicate when we are parsing the items table
    parsing_items = False
    descriptions = []
    comments = []

    while idx < num_lines:
        line = lines[idx].strip()
        if 'Date Issued:' in line:
            # Extract the value after 'Date Issued:'
            parts = line.split('Date Issued:')
            if len(parts) > 1:
                data["Date"] = parts[1].strip()
            else:
                # Handle case where 'Date Issued:' is at the end of the line
                data["Date"] = ""
        elif 'Purchase Order -' in line:
            # Extract the value after 'Purchase Order -'
            parts = line.split('Purchase Order -')
            if len(parts) > 1:
                data["Purchase Order"] = parts[1].strip()
            else:
                data["Purchase Order"] = ""
        elif 'Bill To:' in line:
            # Find the next non-empty line after 'Bill To:'
            idx += 1
            while idx < num_lines and lines[idx].strip() == "":
                idx += 1
            if idx < num_lines:
                data["Company"] = lines[idx].strip()
            else:
                data["Company"] = ""  # No company name found
        elif 'Ship To/ Provide Service At:' in line:
            # Find the next non-empty line after 'Ship To/ Provide Service At:'
            idx += 1
            while idx < num_lines and lines[idx].strip() == "":
                idx += 1
            if idx < num_lines:
                data["Location"] = lines[idx].strip()
            else:
                data["Location"] = ""  # No location found
        elif 'Total ' in line:
            # Extract the value after 'Total '
            parts = line.split('Total ')
            if len(parts) > 1:
                data["Amount"] = parts[1].strip()
            else:
                # Handle case where 'Total ' is at the end of the line
                idx += 1
                while idx < num_lines and lines[idx].strip() == "":
                    idx += 1
                if idx < num_lines:
                    data["Amount"] = lines[idx].strip()
                else:
                    data["Amount"] = ""  # No amount found
        elif 'Item Description' in line and 'Comments' in line:
            # Start parsing the items table
            parsing_items = True
            idx += 1  # Move to the next line after header
            continue  # Skip to the next iteration
        elif parsing_items:
            if 'Delivery to:' in line:
                # Stop parsing items when 'Delivery to:' is encountered
                parsing_items = False
            elif line == "":
                # Skip empty lines
                pass
            else:
                # Parse the item line to extract Description and Comments
                # Assuming the line format:
                # No# Item Number (8) UOM QTY Ext Price ($) Permit Unit # Attch Item Description Comments
                # Example:
                # 1.00 Each 630.00 630.00 | Not Required Yes 30W 57/Troon Electric - Please refer to this PO

                # Split the line by '|' to separate columns
                parts = line.split('|')
                if len(parts) >= 2:
                    # The first part contains up to 'Permit Unit # Attch'
                    # The second part contains 'Item Description Comments'
                    item_info = parts[0].strip().split()
                    # Assuming the last two elements are 'Attch' and the start of 'Item Description'
                    # However, due to OCR inconsistencies, this might need adjustments

                    # Extract Comments (from the first part after '|')
                    comments_part = parts[1].strip()
                    comments.append(comments_part)

                    # Extract Description (from the second part)
                    # Assuming Description is the remainder of the line after Comments
                    # If 'Comments' is already captured, Description is likely to be before 'Comments'
                    # This requires clarity on the exact format; here's an assumption:

                    # Example: '30W 57/Troon Electric - Please refer to this PO'
                    description_part = parts[1].strip()
                    descriptions.append(description_part)
                else:
                    # If '|' is not present, attempt to parse Description and Comments differently
                    # For example, split by multiple spaces or other delimiters
                    # This part may need to be adjusted based on actual data
                    split_line = re.split(r'\s{2,}', line)
                    if len(split_line) >= 2:
                        description = split_line[-2].strip()
                        comment = split_line[-1].strip()
                        descriptions.append(description)
                        comments.append(comment)
        idx += 1

    # Join the descriptions and comments into single strings separated by semicolons
    data["Description"] = "; ".join(descriptions)
    data["Comments"] = "; ".join(comments)

    return data
