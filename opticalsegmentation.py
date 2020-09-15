from PIL import Image
import os
sliceHeight=256
sliceWidth=256
outdir='/home/vikas/Desktop/mouli/sar'
out_name='sar'
img = Image.open('sar.png') # Load image
imageWidth, imageHeight = img.size
print(img.size) # Get image dimensions
left = 0 # Set the left-most edge
upper = 0 # Set the top-most edge
while (left < imageWidth):
	while (upper < imageHeight):
	    # If the bottom and right of the cropping box overruns the image.
	    if (upper + sliceHeight > imageHeight and left + sliceWidth > imageWidth):
             bbox=(left, upper, imageWidth, imageHeight)
	    # If the right of the cropping box overruns the image
	    elif (left + sliceWidth > imageWidth):
             bbox=(left, upper, imageWidth, upper + sliceHeight)
	    # If the bottom of the cropping box overruns the image
	    elif (upper + sliceHeight > imageHeight):
             bbox=(left, upper, left + sliceWidth, imageHeight)
	    # If the entire cropping box is inside the image,
	    # proceed normally.
	    else:
             bbox=(left, upper, left + sliceWidth, upper + sliceHeight)
             working_slice = img.crop(bbox) # Crop image based on created bounds
	    # Save your new cropped image.
	    working_slice.save(os.path.join(outdir, 'slice_' + out_name + '_' + str(upper) + '_' + str(left) + '.png'))
	    upper += sliceHeight # Increment the horizontal position
	left += sliceWidth # Increment the vertical position
	upper = 0


