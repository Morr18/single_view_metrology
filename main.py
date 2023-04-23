import cv2
import numpy as np

# Leer la imagen
# Establecer punto a analizar
img = cv2.imread("image.jpg")

# Definir los puntos de interés en la imagen
corners = cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 10, 0.01, 10)
corners = np.int0(corners)

# Mostrar imagen
#cv2.imshow("antes", img)

# Dibujar los puntos de interés en la imagen
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

# Definir los puntos de objeto correspondientes a los puntos de interés
obj_points = np.zeros((10, 3), np.float32)
obj_points[:, :2] = np.mgrid[0:5, 0:2].T.reshape(-1, 2)


# Determinar los puntos de interes de manera manual con el siguiente codigo o bien con el algoritmo de Harris o canny
object_points = []
def select_points(event, x, y, flags, params):
    
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        object_points.append([x, y, 0])

# Mostrar la imagen y esperar a que se seleccionen los puntos de interés
# Hacer la seleccion de los puntos de aquellos bordes, de preferencia en sentido antihorario
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', select_points)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


pt_A = object_points[0][0:2]
pt_B = object_points[1][0:2]
pt_C = object_points[2][0:2]
pt_D = object_points[3][0:2]


# Here, I have used L2 norm. You can use L1 also.
width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))
 
 
height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts,output_pts)

out = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)

cv2.imshow("out", out)
cv2.waitKey(0)
