import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path) # add name
    data = np.asarray(image)
    return data
    #raise NotImplementedError('You need to implement this function')

# Function to perform KMeans clustering for image quantization
def image_compression(image_np, n_colors):
    pixels = image_np.reshape(-1, 3)            # have to reshape to it is 2D array of pixels and rgb
    # kmeans clustering to get the centers to be for compressed image
    # gets it to n_colors clusters (you specify), and apply kmeans to pixels array
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init="auto").fit(pixels)
    new_colors = kmeans.cluster_centers_.astype(np.uint8) # the colors of the new image are the cluster centers
    # labels tells you for each pixel which cluster it belongs to (array) (basiclaly gives you the label fo each point, aka which cluster its in)
    labels = kmeans.labels_
    # use the centers of clusters from kmeans as the pixel and reshape
    comp_array = new_colors[labels].reshape(image_np.shape)
    # dont need to turn back to image comp_image = Image.fromarray(comp_array)
    return comp_array
    #raise NotImplementedError('You need to implement this function')

# Function to concatenate and save the original and quantized images side by side
def save_result(original_image_np, quantized_image_np, output_path):
    # Convert NumPy arrays back to PIL images
    original_image = Image.fromarray(original_image_np)
    quantized_image = Image.fromarray(quantized_image_np)
    
    # Get dimensions
    width, height = original_image.size
    
    # Create a new image that will hold both the original and quantized images side by side
    combined_image = Image.new('RGB', (width * 2, height))
    
    # Paste original and quantized images side by side
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(quantized_image, (width, 0))
    
    # Save the combined image
    combined_image.save(output_path)

if __name__ == "__main__":
    # Load and process the image
    image_path = 'flower.png'  
    output_path = 'compressed_image.png'  
    image_np = load_image(image_path)

    # Perform image quantization using KMeans
    n_colors = 8  # Number of colors to reduce the image to, you may change this to experiment
    quantized_image_np = image_compression(image_np, n_colors)

    # Save the original and quantized images side by side
    save_result(image_np, quantized_image_np, output_path)
