import math
import numpy
import cv2
import scipy.ndimage
import skimage.morphology
from skimage.filters import gabor_kernel

#Section A: Acquire

def acquire_from_file(file_path, view=False):
    fingerprint = cv2.imread(file_path)

    if view:
        cv2.imshow('press any key', fingerprint)
        cv2.waitKey(0)

    print('[INFO] Acquired fingerprint from file.')
    return fingerprint

#Section B: Enhance

FINGERPRINT_HEIGHT = 352

FINGERPRINT_BLOCK = 16

FINGERPRINT_MASK_TRSH = 0.25

RIDGE_ORIENTATION_STEP = numpy.pi / 16
RIDGE_ORIENTATIONS = numpy.arange(-numpy.pi, numpy.pi + RIDGE_ORIENTATION_STEP, RIDGE_ORIENTATION_STEP)

WAVELENGTH_RATIO = 0.25

GABOR_OUTPUT_BIN_TRSH = -0.2

def __rotate_and_crop(image, rad_angle):
    h, w = image.shape

    degree_angle = 360.0 - (180.0 * rad_angle / numpy.pi)
    rotated = scipy.ndimage.rotate(image, degree_angle, reshape=False)

    crop_size = int(h / numpy.sqrt(2))
    crop_start = int((h - crop_size) / 2.0)

    rotated = rotated[crop_start: crop_start + crop_size, crop_start: crop_start + crop_size]
    return rotated

def _01_preprocess(fingerprint, output_height, dark_ridges=True, view=False):
    if len(fingerprint.shape) > 2 and fingerprint.shape[2] > 1:
        fingerprint = cv2.cvtColor(fingerprint, cv2.COLOR_BGR2GRAY)

    aspect_ratio = float(fingerprint.shape[0]) / fingerprint.shape[1]
    width = int(round(output_height / aspect_ratio))
    fingerprint = cv2.resize(fingerprint, (width, output_height))

    if not dark_ridges:
        fingerprint = abs(255 - fingerprint)

    fingerprint = cv2.equalizeHist(fingerprint, fingerprint)

    if view:
        cv2.imshow('Preprocessing, press any key.', fingerprint)
        cv2.waitKey(0)

    print('[INFO] Preprocessed fingerprint.')
    return fingerprint

def _02_segment(fingerprint, block_size, std_threshold, view=False):
    h, w = fingerprint.shape

    fingerprint = (fingerprint - numpy.mean(fingerprint)) / numpy.std(fingerprint)

    mask = numpy.zeros((h, w), numpy.uint8)

    block_step = int(block_size / 2.0)
    for row in range(h):
        for col in range(w):
            block = fingerprint[max(0, row - block_step):min(row + block_step + 1, h),
                    max(0, col - block_step):min(col + block_step + 1, w)]

            if numpy.std(block) > std_threshold:
                mask[row, col] = 255

    masked_values = fingerprint[mask > 0]
    fingerprint = (fingerprint - numpy.mean(masked_values)) / numpy.std(masked_values)
    fingerprint = cv2.bitwise_and(fingerprint, fingerprint, mask=mask)

    if view:
        img = fingerprint.copy()
        img = cv2.normalize(fingerprint, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow('Segmentation, press any key.', img)
        cv2.waitKey(0)

    print('[INFO] Segmented fingerprint.')
    return fingerprint, mask


def _03_compute_orientations(fingerprint, mask, block_size, view=False):
    h, w = fingerprint.shape

    y_gradient, x_gradient = numpy.gradient(fingerprint)

    orientations = numpy.arctan2(y_gradient, x_gradient)
    orientations = cv2.bitwise_and(orientations, orientations, mask=mask)

    magnitudes = numpy.sqrt(y_gradient ** 2 + x_gradient ** 2)
    magnitudes = cv2.bitwise_and(magnitudes, magnitudes, mask=mask)

    discret_orientations = numpy.zeros(orientations.shape, dtype=numpy.float32)
    block_step = int(block_size / 2.0)
    for row in range(h):
        for col in range(w):
            if mask[row, col] > 0:
                ori_block = orientations[max(0, row - block_step):min(row + block_step + 1, h),
                            max(0, col - block_step):min(col + block_step + 1, w)]
                mag_block = magnitudes[max(0, row - block_step):min(row + block_step + 1, h),
                            max(0, col - block_step):min(col + block_step + 1, w)]

                useful_magnitudes = numpy.where(mag_block > numpy.mean(mag_block))
                freqs, values = numpy.histogram(ori_block[useful_magnitudes], bins=RIDGE_ORIENTATIONS)

                best_value = numpy.mean(values[numpy.where(freqs == numpy.max(freqs))])
                orientation_index = int(round(best_value / RIDGE_ORIENTATION_STEP))
                discret_orientations[row, col] = RIDGE_ORIENTATIONS[orientation_index]

    discret_orientations = cv2.bitwise_and(discret_orientations, discret_orientations, mask=mask)

    if view:
        img = x_gradient.copy()
        img = cv2.normalize(x_gradient, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow('Orientation, x gradient, press any key.', img)
        cv2.waitKey(0)

        img = y_gradient.copy()
        img = cv2.normalize(y_gradient, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow('Orientation, y gradient, press any key.', img)
        cv2.waitKey(0)

        plot_step = 8

        img = fingerprint.copy()
        img = cv2.normalize(fingerprint, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        mag_enhance = 5.0
        start_pixel = int(plot_step / 2.0)
        for row in range(start_pixel, h, plot_step):
            for col in range(start_pixel, w, plot_step):
                angle = discret_orientations[row, col]
                magnitude = magnitudes[row, col] * mag_enhance

                if magnitude > 0:
                    delta_x = int(round(math.cos(angle) * magnitude))
                    delta_y = int(round(math.sin(angle) * magnitude))

                    cv2.line(img, (col, row), (col + delta_x, row + delta_y), (0, 255, 0), 1)

        cv2.imshow('Orientation, press any key.', img)
        cv2.waitKey(0)

    print('[INFO] Computed ridge orientations.')
    return discret_orientations, magnitudes


def _04_compute_ridge_frequency(fingerprint, mask, orientations, block_size, view=False):
    frequencies = []

    h, w = fingerprint.shape

    block_step = int(block_size / 2.0)
    for row in range(h):
        for col in range(w):
            if mask[row, col] > 0:
                block = fingerprint[max(0, row - block_step):min(row + block_step + 1, h),
                        max(0, col - block_step):min(col + block_step + 1, w)]

                rot_block = __rotate_and_crop(block, -orientations[row, col])

                if view:
                    img = rot_block.copy()
                    img = cv2.normalize(rot_block, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                    cv2.imshow('block', img)
                    cv2.waitKey(120)

                ridge_proj = numpy.sum(rot_block, axis=0)
                ridge_peaks = numpy.zeros(ridge_proj.shape)
                ridge_peaks[numpy.where(ridge_proj > numpy.mean(ridge_proj))] = 1

                ridge_count = 0

                is_ridge = False
                for i in range(len(ridge_peaks)):
                    if ridge_peaks[i] == 1 and not is_ridge:
                        ridge_count = ridge_count + 1
                        is_ridge = True

                    elif ridge_peaks[i] == 0 and is_ridge:
                        ridge_count = ridge_count + 1
                        is_ridge = False

                frequencies.append(0.5 * ridge_count / len(ridge_peaks))

    print('[INFO] Computed ridge frequency.')
    if len(frequencies) > 0:
        return numpy.mean(frequencies)
    else:
        return 0


def _05_apply_gabor_filter(fingerprint, mask, orientations, ridge_frequency, std_wavelength_ratio, view=False):
    output = numpy.zeros(fingerprint.shape)

    h, w = fingerprint.shape

    fingerprint_filters = {}

    filter_std = std_wavelength_ratio * 1.0 / ridge_frequency
    for orientation in numpy.unique(orientations):
        kernel = numpy.real(gabor_kernel(ridge_frequency, orientation, sigma_x=filter_std, sigma_y=filter_std))
        fingerprint_filters[orientation] = scipy.ndimage.convolve(fingerprint, kernel)

    for row in range(h):
        for col in range(w):
            if mask[row, col] > 0:
                key_orientation = orientations[row, col]
                output[row, col] = fingerprint_filters[key_orientation][row, col]

    output = (output < GABOR_OUTPUT_BIN_TRSH).astype(numpy.uint8) * 255
    output = cv2.erode(output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

    if view:
        cv2.imshow('Filtering, press any key.', output)
        cv2.waitKey(0)

    print('[INFO] Applied Gabor filters.')
    return output


def _06_skeletonize(fingerprint, view=False):
    fingerprint = skimage.morphology.skeletonize(fingerprint / 255).astype(numpy.uint8) * 255

    if view:
        cv2.imshow('Skeletonization, press any key.', fingerprint)
        cv2.waitKey(0)

    print('[INFO] Skeletonized ridges.')
    return fingerprint


def enhance(fingerprint, dark_ridges=True, view=False):
    pp_fingerprint = _01_preprocess(fingerprint, FINGERPRINT_HEIGHT, dark_ridges, view=view)

    en_fingerprint, mask = _02_segment(pp_fingerprint, FINGERPRINT_BLOCK, FINGERPRINT_MASK_TRSH, view=view)

    orientations, magnitudes = _03_compute_orientations(en_fingerprint, mask, FINGERPRINT_BLOCK, view=view)

    ridge_freq = _04_compute_ridge_frequency(en_fingerprint, mask, orientations, FINGERPRINT_BLOCK)

    en_fingerprint = _05_apply_gabor_filter(en_fingerprint, mask, orientations, ridge_freq, WAVELENGTH_RATIO, view=view)

    en_fingerprint = _06_skeletonize(en_fingerprint, view=view)

    print('[INFO] Enhanced fingerprint.')
    return pp_fingerprint, en_fingerprint, mask

#Section C: Describe

MINUT_ORIENT_BLOCK_SIZE = 7
MIN_MINUTIAE_DIST = 5
MIN_RIDGE_LENGTH = 10
RIDGE_END_ANGLE_TOLER = numpy.pi / 8.0
MIN_MINUT_MASK_DIST = 20


def __draw_minutiae(fingerprint, ridge_endings, ridge_bifurcations, msg):
    mag = 5.0

    img = (fingerprint > 0).astype(numpy.uint8) * 255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for ridge_ending in ridge_endings:
        p = (int(ridge_ending[0]), int(ridge_ending[1]))
        cv2.rectangle(img, (p[0] - 1, p[1] - 1), (p[0] + 2, p[1] + 2), (0, 0, 255), 1)

        delta_x = int(round(math.cos(ridge_ending[2]) * mag))
        delta_y = int(round(math.sin(ridge_ending[2]) * mag))
        cv2.line(img, (p[0], p[1]), (p[0] + delta_x, p[1] + delta_y), (0, 255, 255), 1)

    for bifurcation in ridge_bifurcations:
        p = (int(bifurcation[0]), int(bifurcation[1]))
        cv2.rectangle(img, (p[0] - 1, p[1] - 1), (p[0] + 2, p[1] + 2), (0, 255, 0), 1)

        delta_x = int(round(math.cos(bifurcation[2]) * mag))
        delta_y = int(round(math.sin(bifurcation[2]) * mag))
        cv2.line(img, (p[0], p[1]), (p[0] + delta_x, p[1] + delta_y), (0, 255, 255), 1)

    cv2.imshow(msg + ' minutiae, press any key.', img)
    cv2.waitKey(0)


def __compute_minutiae_angle(fingerprint, position, block_size, is_ridge_ending):
    h, w = fingerprint.shape

    block_step = int(block_size / 2)
    block = fingerprint[max(0, position[1] - block_step):min(position[1] + block_step + 1, h),
            max(0, position[0] - block_step):min(position[0] + block_step + 1, w)]
    block_h, block_w = block.shape
    block_center = (int(block_w / 2), int(block_h / 2))

    border_points = []
    for i in range(block_h):
        for j in range(block_w):
            if (i == 0 or i == block_h - 1 or j == 0 or j == block_w - 1) and block[i, j] > 0:
                border_points.append((j, i))

    if is_ridge_ending:
        if len(border_points) < 1:
            return None

        closest_point = None
        closest_distance = float('inf')
        for p in border_points:
            dist = math.sqrt((p[0] - block_center[0]) ** 2 + (p[1] - block_center[1]) ** 2)
            if dist < closest_distance:
                closest_point = p
                closest_distance = dist

        return math.atan2(closest_point[1] - block_center[1], closest_point[0] - block_center[0])

    else:
        if len(border_points) != 3:
            return None

        closest_points = None
        closest_distance = float('inf')
        for i in range(len(border_points) - 1):
            for j in range(i + 1, len(border_points)):
                dist = math.sqrt(
                    (border_points[i][0] - border_points[j][0]) ** 2 + (border_points[i][1] - border_points[j][1]) ** 2)
                if dist < closest_distance:
                    closest_points = [border_points[i], border_points[j]]
                    closest_distance = dist

        mid_point = numpy.mean(closest_points, axis=0)
        return math.atan2(mid_point[1] - block_center[1], mid_point[0] - block_center[0])


def _01_detect_minutiae(fingerprint, mask, block_size, view=False):
    ridge_endings = []
    ridge_bifurcations = []

    fingerprint = (fingerprint > 0).astype(numpy.uint8)

    h, w = fingerprint.shape

    for row in range(1, h - 1):
        for col in range(1, w - 1):
            if mask[row, col] > 0:
                if fingerprint[row, col] == 1:
                    block = fingerprint[row - 1: row + 2, col - 1: col + 2]

                    ridge_count = numpy.sum(block)

                    if ridge_count < 3:
                        angle = __compute_minutiae_angle(fingerprint, (col, row), block_size, is_ridge_ending=True)
                        if angle is not None:
                            ridge_endings.append((col, row, angle))

                    elif ridge_count > 3:
                        angle = __compute_minutiae_angle(fingerprint, (col, row), block_size, is_ridge_ending=False)
                        if angle is not None:
                            ridge_bifurcations.append((col, row, angle))

    if view:
        __draw_minutiae(fingerprint, ridge_endings, ridge_bifurcations, 'All')

    print('[INFO] Detected minutiae.')
    return ridge_endings, ridge_bifurcations


def _02_remove_false_positive_minutiae(fingerprint, mask, ridge_endings, ridge_bifurcations,
                                       min_minutiae_dist, min_ridge_length, ridge_ending_angle_tol, min_mask_dist,
                                       view=False):
    h, w = fingerprint.shape

    ridge_ending_count = len(ridge_endings)
    bifurcation_count = len(ridge_bifurcations)

    good_ridge_endings = [True] * ridge_ending_count
    good_bifurcations = [True] * bifurcation_count


    for i in range(0, ridge_ending_count - 1):
        for j in range(i + 1, ridge_ending_count):
            x0, y0, a0 = ridge_endings[i]
            x1, y1, a1 = ridge_endings[j]

            dist = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
            if dist < min_minutiae_dist:
                good_ridge_endings[i] = good_ridge_endings[j] = False

    for i in range(0, ridge_ending_count - 1):
        for j in range(i + 1, ridge_ending_count):
            x0, y0, a0 = ridge_endings[i]
            x1, y1, a1 = ridge_endings[j]

            if a0 < 0.0:
                a0 = 2.0 * numpy.pi + a0

            if a1 < 0.0:
                a1 = 2.0 * numpy.pi + a1

            dist = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
            if dist < min_ridge_length:
                a01 = math.atan2(y0 - y1, x0 - x1)
                if a01 < 0.0:
                    a01 = 2.0 * numpy.pi + a01

                if abs(a01 - a0) < ridge_ending_angle_tol or abs(a01 - a1) < ridge_ending_angle_tol:
                    if numpy.pi - ridge_ending_angle_tol < abs(a0 - a1) < numpy.pi + ridge_ending_angle_tol:
                        good_ridge_endings[i] = good_ridge_endings[j] = False

    for i in range(ridge_ending_count):
        for j in range(bifurcation_count):
            x0, y0, _ = ridge_endings[i]
            x1, y1, _ = ridge_bifurcations[j]

            dist = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
            if dist < min_minutiae_dist:
                good_ridge_endings[i] = good_bifurcations[j] = False

    for i in range(0, ridge_ending_count):
        if good_ridge_endings[i]:
            x0, y0, _ = ridge_endings[i]

            if x0 - min_mask_dist < 0 or y0 - min_mask_dist < 0 or \
                    x0 + min_mask_dist + 1 > w or y0 + min_mask_dist + 1 > h:
                good_ridge_endings[i] = False

            else:
                mask_block = mask[y0 - min_mask_dist:y0 + min_mask_dist + 1, x0 - min_mask_dist:x0 + min_mask_dist + 1]

                if numpy.min(mask_block) == 0:
                    good_ridge_endings[i] = False

    for i in range(0, bifurcation_count):
        if good_bifurcations[i]:
            x0, y0, _ = ridge_bifurcations[i]

            if x0 - min_mask_dist < 0 or y0 - min_mask_dist < 0 or \
                    x0 + min_mask_dist + 1 > w or y0 + min_mask_dist + 1 > h:
                good_ridge_endings[i] = False

            else:
                mask_block = mask[y0 - min_mask_dist:y0 + min_mask_dist + 1, x0 - min_mask_dist:x0 + min_mask_dist + 1]

                if numpy.min(mask_block) == 0:
                    good_ridge_endings[i] = False

                if numpy.min(mask_block) == 0:
                    good_bifurcations[i] = False

    ridge_endings = numpy.array(ridge_endings)[numpy.where(good_ridge_endings)]
    ridge_bifurcations = numpy.array(ridge_bifurcations)[numpy.where(good_bifurcations)]

    if view:
        __draw_minutiae(fingerprint, ridge_endings, ridge_bifurcations, 'Cleaned')

    print('[INFO] Removed bad-quality minutiae.')
    return ridge_endings, ridge_bifurcations


def describe(enhanced_fingerprint, mask, view=False):
    ridge_endings, bifurcations = _01_detect_minutiae(enhanced_fingerprint, mask, MINUT_ORIENT_BLOCK_SIZE, view=view)

    ridge_endings, bifurcations = _02_remove_false_positive_minutiae(enhanced_fingerprint, mask,
                                                                     ridge_endings, bifurcations,
                                                                     MIN_MINUTIAE_DIST, MIN_RIDGE_LENGTH,
                                                                     RIDGE_END_ANGLE_TOLER, MIN_MINUT_MASK_DIST,
                                                                     view=view)

    return ridge_endings, bifurcations

#Section D: Match

HOUGH_SCALE_RANGE = [1.0]


HOUGH_ROTATION_STEP = numpy.pi / 8.0
HOUGH_ROTATION_RANGE = numpy.arange(-numpy.pi / 2.0, numpy.pi / 2.0, HOUGH_ROTATION_STEP)


HOUGH_TRANSLATION_OVERLAY_RATE = 0.5
HOUGH_TRANSLATION_STEP = 10


DIST_TRSH = 10
ANGLE_TRSH = numpy.pi / 8.0


def __compare(minutiae_1, minutiae_2, dist_threshold, angle_threshold):
    dist = math.sqrt((minutiae_1[0] - minutiae_2[0]) ** 2 + (minutiae_1[1] - minutiae_2[1]) ** 2)
    if dist > dist_threshold:
        return float('inf')

    a1 = minutiae_1[2]
    if a1 < 0.0:
        a1 = 2.0 * numpy.pi + a1

    a2 = minutiae_2[2]
    if a2 < 0.0:
        a2 = 2.0 * numpy.pi + a2

    angle_diff = abs(a1 - a2)
    if angle_diff > angle_threshold:
        return float('inf')

    return (dist / dist_threshold + angle_diff / angle_threshold) / 2.0


def _compute_matches(minutiae_1_points, minutiae_1_angles, minutiae_1_types,
                     minutiae_2_points, minutiae_2_angles, minutiae_2_types,
                     x_scale, y_scale, rotation, translation_overlay_rate, translation_step,
                     dist_threshold, angle_threshold):
    scale_matrix = numpy.zeros((3, 3), dtype=numpy.float32)
    scale_matrix[0, 0] = x_scale
    scale_matrix[1, 1] = y_scale
    scale_matrix[2, 2] = 1.0
    minutiae_2_points = cv2.perspectiveTransform(numpy.float32([minutiae_2_points]), scale_matrix)[0]

    if rotation < 0.0:
        rotation = 2.0 * numpy.pi + rotation

    sine = math.sin(rotation)
    cosine = math.cos(rotation)

    rotation_matrix = numpy.zeros((3, 3))
    rotation_matrix[0, 0] = cosine
    rotation_matrix[0, 1] = -sine
    rotation_matrix[1, 0] = sine
    rotation_matrix[1, 1] = cosine
    rotation_matrix[2, 2] = 1.0
    minutiae_2_points = cv2.perspectiveTransform(numpy.float32([minutiae_2_points]), rotation_matrix)[0]

    minutiae_2_angles = minutiae_2_angles.copy()
    for i in range(len(minutiae_2_angles)):
        angle = minutiae_2_angles[i]
        if angle < 0.0:
            angle = 2.0 * numpy.pi + angle

        new_angle = angle + rotation
        if new_angle > numpy.pi:
            new_angle = new_angle - 2.0 * numpy.pi

        minutiae_2_angles[i] = new_angle

    minutiae_1_points = minutiae_1_points - [numpy.min(minutiae_1_points[:, 0]), numpy.min(minutiae_1_points[:, 1])]
    minutiae_2_points = minutiae_2_points - [numpy.max(minutiae_2_points[:, 0]), numpy.max(minutiae_2_points[:, 1])]

    minutiae_1_corner_1 = numpy.array([numpy.min(minutiae_1_points[:, 0]), numpy.min(minutiae_1_points[:, 1])],
                                      dtype=int)
    minutiae_1_corner_2 = numpy.array([numpy.max(minutiae_1_points[:, 0]), numpy.max(minutiae_1_points[:, 1])],
                                      dtype=int)

    minutiae_1_w, minutiae_1_h = minutiae_1_corner_2 - minutiae_1_corner_1
    minutiae_1_x_offset = int(round((1.0 - translation_overlay_rate) * minutiae_1_w / 2.0))
    minutiae_1_y_offset = int(round((1.0 - translation_overlay_rate) * minutiae_1_h / 2.0))

    minutiae_2_corner_1 = numpy.array([numpy.min(minutiae_2_points[:, 0]), numpy.min(minutiae_2_points[:, 1])],
                                      dtype=int)
    minutiae_2_corner_2 = numpy.array([numpy.max(minutiae_2_points[:, 0]), numpy.max(minutiae_2_points[:, 1])],
                                      dtype=int)

    minutiae_2_w, minutiae_2_h = minutiae_2_corner_2 - minutiae_2_corner_1
    minutiae_2_x_offset = int(round((1.0 - translation_overlay_rate) * minutiae_2_w / 2.0))
    minutiae_2_y_offset = int(round((1.0 - translation_overlay_rate) * minutiae_2_h / 2.0))

    start_x = minutiae_1_x_offset + minutiae_2_x_offset
    stop_x = start_x + minutiae_1_w + minutiae_2_w - minutiae_1_x_offset - minutiae_2_x_offset
    start_y = minutiae_1_y_offset + minutiae_2_y_offset
    stop_y = start_y + minutiae_1_h + minutiae_2_h - minutiae_1_y_offset - minutiae_2_y_offset

    best_matches = []

    for x_translation in range(start_x, stop_x, translation_step):
        for y_translation in range(start_y, stop_y, translation_step):
            minutiae_2_points_transl = minutiae_2_points + [x_translation, y_translation]

            matches = []
            already_matched = []
            for i in range(minutiae_1_points.shape[0]):
                current_match = None
                current_match_dist = float('inf')

                for j in range(minutiae_2_points_transl.shape[0]):
                    if j not in already_matched and minutiae_1_types[i] == minutiae_2_types[j] and \
                            minutiae_2_points_transl[j][0] > 0.0 and minutiae_2_points_transl[j][1] > 0.0:
                        dist = __compare((minutiae_1_points[i][0], minutiae_1_points[i][1], minutiae_1_angles[i]),
                                         (minutiae_2_points_transl[j][0], minutiae_2_points_transl[j][1],
                                          minutiae_2_angles[j]), dist_threshold, angle_threshold)
                        if dist < current_match_dist:
                            current_match = j
                            current_match_dist = dist

                if current_match is not None:
                    matches.append((i, current_match))
                    already_matched.append(current_match)

            if len(best_matches) < len(matches):
                best_matches = matches

    return best_matches


def _draw_matches(fingerprint_1, fingerprint_2, matches,
                  ridge_endings_1, bifurcations_1,
                  ridge_endings_2, bifurcations_2):
    mag = 5.0

    h1, w1 = fingerprint_1.shape
    h2, w2 = fingerprint_2.shape

    output_image = numpy.zeros((max(h1, h2), w1 + w2), dtype=numpy.uint8)
    output_image[0:h1, 0:w1] = (fingerprint_1 > 0).astype(numpy.uint8) * 255
    output_image[0:h2, w1:w1 + w2] = (fingerprint_2 > 0).astype(numpy.uint8) * 255
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)

    for ridge_ending in ridge_endings_1:
        p = (int(ridge_ending[0]), int(ridge_ending[1]))
        cv2.rectangle(output_image, (p[0] - 1, p[1] - 1), (p[0] + 2, p[1] + 2), (0, 0, 255), 1)

        delta_x = int(round(math.cos(ridge_ending[2]) * mag))
        delta_y = int(round(math.sin(ridge_ending[2]) * mag))
        cv2.line(output_image, (p[0], p[1]), (p[0] + delta_x, p[1] + delta_y), (0, 255, 255), 1)

    for ridge_ending in ridge_endings_2:
        p = (int(ridge_ending[0] + w1), int(ridge_ending[1]))
        cv2.rectangle(output_image, (p[0] - 1, p[1] - 1), (p[0] + 2, p[1] + 2), (0, 0, 255), 1)

        delta_x = int(round(math.cos(ridge_ending[2]) * mag))
        delta_y = int(round(math.sin(ridge_ending[2]) * mag))
        cv2.line(output_image, (p[0] + w1, p[1]), (p[0] + delta_x + w1, p[1] + delta_y), (0, 255, 255), 1)

    for bifurcation in bifurcations_1:
        p = (int(bifurcation[0]), int(bifurcation[1]))
        cv2.rectangle(output_image, (p[0] - 1, p[1] - 1), (p[0] + 2, p[1] + 2), (0, 255, 0), 1)

        delta_x = int(round(math.cos(bifurcation[2]) * mag))
        delta_y = int(round(math.sin(bifurcation[2]) * mag))
        cv2.line(output_image, (p[0], p[1]), (p[0] + delta_x, p[1] + delta_y), (0, 255, 255), 1)

    for bifurcation in bifurcations_2:
        p = (int(bifurcation[0] + w1), int(bifurcation[1]))
        cv2.rectangle(output_image, (p[0] - 1, p[1] - 1), (p[0] + 2, p[1] + 2), (0, 255, 0), 1)

        delta_x = int(round(math.cos(bifurcation[2]) * mag))
        delta_y = int(round(math.sin(bifurcation[2]) * mag))
        cv2.line(output_image, (p[0] + w1, p[1]), (p[0] + delta_x + w1, p[1] + delta_y), (0, 255, 255), 1)


    for m in matches[0]:
        x0 = int(m[0][0])
        y0 = int(m[0][1])
        a0 = m[0][2]
        delta_x0 = int(round(math.sin(a0) * mag))
        delta_y0 = int(round(math.cos(a0) * mag))

        x1 = int(m[1][0])
        y1 = int(m[1][1])
        a1 = m[1][2]
        delta_x1 = int(round(math.sin(a1) * mag))
        delta_y1 = int(round(math.cos(a1) * mag))

        cv2.rectangle(output_image, (x0 - 1, y0 - 1), (x0 + 2, y0 + 2), (0, 0, 255), 1)
        cv2.rectangle(output_image, (x1 - 1 + w1, y1 - 1), (x1 + 2 + w1, y1 + 2), (0, 0, 255), 1)
        cv2.line(output_image, (x0, y0), (x0 + delta_x0, y0 + delta_y0), (0, 255, 255), 1)
        cv2.line(output_image, (x1 + w1, y1), (x1 + w1 + delta_x1, y1 + delta_y1), (0, 255, 255), 1)
        cv2.line(output_image, (x0, y0), (x1 + w1, y1), (0, 255, 255), 1)


    for m in matches[1]:
        x0 = int(m[0][0])
        y0 = int(m[0][1])
        a0 = m[0][2]
        delta_x0 = int(round(math.sin(a0) * mag))
        delta_y0 = int(round(math.cos(a0) * mag))

        x1 = int(m[1][0])
        y1 = int(m[1][1])
        a1 = m[1][2]
        delta_x1 = int(round(math.sin(a1) * mag))
        delta_y1 = int(round(math.cos(a1) * mag))

        cv2.rectangle(output_image, (x0 - 1, y0 - 1), (x0 + 2, y0 + 2), (0, 255, 0), 1)
        cv2.rectangle(output_image, (x1 - 1 + w1, y1 - 1), (x1 + 2 + w1, y1 + 2), (0, 255, 0), 1)
        cv2.line(output_image, (x0, y0), (x0 + delta_x0, y0 + delta_y0), (0, 255, 255), 1)
        cv2.line(output_image, (x1 + w1, y1), (x1 + w1 + delta_x1, y1 + delta_y1), (0, 255, 255), 1)
        cv2.line(output_image, (x0, y0), (x1 + w1, y1), (0, 255, 255), 1)

    cv2.imshow('',output_image)
    cv2.waitKey(0)


def _01_hough_transform(ridge_endings_1, ridge_bifurcations_1, ridge_endings_2, ridge_bifurcations_2,
                        scale_range, rotation_range, translation_overlay_rate, translation_step,
                        dist_threshold, angle_threshold):

    minutiae_set_1 = numpy.concatenate((ridge_endings_1, ridge_bifurcations_1), axis=0)
    minutiae_set_2 = numpy.concatenate((ridge_endings_2, ridge_bifurcations_2), axis=0)
    if len(minutiae_set_1) == 0 or len(minutiae_set_2) == 0:
        return [], []

    minutiae_1_points = numpy.array([[minutiae_set_1[0][0], minutiae_set_1[0][1]]])
    minutiae_1_angles = [minutiae_set_1[0][2]]
    minutiae_1_types = [True] * len(ridge_endings_1) + [False] * len(ridge_bifurcations_1)
    for i in range(1, len(minutiae_set_1)):
        minutiae_1_points = numpy.append(minutiae_1_points, [[minutiae_set_1[i][0], minutiae_set_1[i][1]]], 0)
        minutiae_1_angles.append(minutiae_set_1[i][2])

    minutiae_2_points = numpy.array([[minutiae_set_2[0][0], minutiae_set_2[0][1]]])
    minutiae_2_angles = [minutiae_set_2[0][2]]
    minutiae_2_types = [True] * len(ridge_endings_2) + [False] * len(ridge_bifurcations_2)
    for i in range(1, len(minutiae_set_2)):
        minutiae_2_points = numpy.append(minutiae_2_points, [[minutiae_set_2[i][0], minutiae_set_2[i][1]]], 0)
        minutiae_2_angles.append(minutiae_set_2[i][2])

    best_matches = []
    best_config = None

    for x_scale in scale_range:
        for y_scale in scale_range:
            for rotation in rotation_range:
                matches = _compute_matches(minutiae_1_points, minutiae_1_angles, minutiae_1_types,
                                           minutiae_2_points, minutiae_2_angles, minutiae_2_types,
                                           x_scale, y_scale, rotation, translation_overlay_rate, translation_step,
                                           dist_threshold, angle_threshold)

                print('[INFO] Hough transform at', str([x_scale, y_scale, rotation]) + ':', len(matches), 'matches.')

                if len(best_matches) < len(matches):
                    best_matches = matches
                    best_config = [x_scale, y_scale, rotation]

    print('[INFO] Best Hough with:', len(best_matches), 'matches, at:', str(best_config) + '.')

    ridge_ending_matches = []
    bifurcation_matches = []
    for m in best_matches:
        if minutiae_1_types[m[0]]:
            ridge_ending_matches.append(((minutiae_set_1[m[0]][0], minutiae_set_1[m[0]][1], minutiae_1_angles[m[0]]),
                                         (minutiae_set_2[m[1]][0], minutiae_set_2[m[1]][1], minutiae_2_angles[m[1]])))

        else:
            bifurcation_matches.append(((minutiae_set_1[m[0]][0], minutiae_set_1[m[0]][1], minutiae_1_angles[m[0]]),
                                        (minutiae_set_2[m[1]][0], minutiae_set_2[m[1]][1], minutiae_2_angles[m[1]])))

    return ridge_ending_matches, bifurcation_matches


def match(fingerprint_1, ridge_endings_1, ridge_bifurcations_1, fingerprint_2, ridge_endings_2, ridge_bifurcations_2,
          view=False):

    matches = _01_hough_transform(ridge_endings_1, ridge_bifurcations_1, ridge_endings_2, ridge_bifurcations_2,
                                  HOUGH_SCALE_RANGE, HOUGH_ROTATION_RANGE,
                                  HOUGH_TRANSLATION_OVERLAY_RATE, HOUGH_TRANSLATION_STEP,
                                  DIST_TRSH, ANGLE_TRSH)

    if view:
        _draw_matches(fingerprint_1, fingerprint_2, matches, ridge_endings_1, ridge_bifurcations_1, ridge_endings_2,
                      ridge_bifurcations_2)

    return matches



fingeprint_filepath_1 = 'dataset/10__M_Left_index_finger_Zcut.BMP'
fingeprint_filepath_2 = 'dataset/10__M_Left_index_finger_Zcut.BMP'

fingerprint_1 = acquire_from_file(fingeprint_filepath_1, view=False)
fingerprint_2 = acquire_from_file(fingeprint_filepath_2, view=False)

pp_fingerprint_1, en_fingerprint_1, mask_1 = enhance(fingerprint_1, dark_ridges=False, view=False)
pp_fingerprint_2, en_fingerprint_2, mask_2 = enhance(fingerprint_2, dark_ridges=False, view=False)

ridge_endings_1, bifurcations_1 = describe(en_fingerprint_1, mask_1, view=False)
ridge_endings_2, bifurcations_2 = describe(en_fingerprint_2, mask_2, view=False)

match = match(en_fingerprint_1, ridge_endings_1, bifurcations_1, en_fingerprint_2, ridge_endings_2, bifurcations_2, view=True)

#match score calculations
minutiae_1 = len(ridge_endings_1)+len(bifurcations_1)
minutiae_2 = len(ridge_endings_2)+len(bifurcations_2)
match_score = ((minutiae_1 + minutiae_2)/2)
print(minutiae_1)
print(minutiae_2)
print("Match score: ")
print(match_score)