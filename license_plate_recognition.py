# Reading the data
images = []
for i in range(1,21):
    images.append(cv2.imread('{}.jpg'.format(i)))

# Creating a function which takes the raw image as argument and returns image where the number plate is highlighted.
def extract_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 15, 50, 70)
    only_edges = cv2.Canny(blur, 30, 200)
    
    contours, new = cv2.findContours(only_edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse=True)[:30]

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        no_of_edges = cv2.approxPolyDP(c, 0.01 * perimeter, True)
        if len(no_of_edges) == 4:
            x, y, w, h = cv2.boundingRect(c)
            #if h > 25 and w/h > 3.5 and w/h < 5:
            image_copy = image.copy()
            image_copy = cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0,255,0), 3)
            return image_copy
    return image

#calling the function and printing the highlighted images
plt.figure(figsize=(20, 100))
for i in range(len(images)):
    plt.subplot(10,2,i+1)
    plt.title('{}.jpg'.format(i+1))
    plt.imshow(extract_license_plate(images[i]), cmap='gray')
plt.show()
