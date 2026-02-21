import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_filled_lines_from_contours(img, contours):
    black_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if contours is not None:
        cv2.drawContours(black_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return black_image

def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video en la ruta: {video_path}")
            return
    except Exception as e:
        print(f"Error al abrir el video: {e}")
        return

    cv2.namedWindow("Asistencia de manejo", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Fin del video o no se puede leer más frames.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 50, 150)

        kernel = np.ones((6, 6), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        height, width = dilated_edges.shape
        region_of_interest_vertices = [
            (0.1, height),
            (width // 2, height // 1.77),
            (width, height),
        ]
        cropped_edges = region_of_interest(dilated_edges, np.array([region_of_interest_vertices], np.int32))

        contours, _ = cv2.findContours(cropped_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        image_with_filled_lines = draw_filled_lines_from_contours(frame, contours)

        cv2.imshow("Asistencia de manejo", image_with_filled_lines)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = 'lineas.mp4'  
process_video(video_path)
