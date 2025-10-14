import cv2
import numpy as np


def undistort(img, K, D, DIM):
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def undistort_upd(img, K, D, DIM, balance=0.0, dim2=None, dim3=None):
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def calibrate(imgs, CHECKERBOARD=(8, 6)):
    subpix_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.1,
    )
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)

    _img_shape = None
    objpoints = []
    imgpoints = []

    for img in imgs:
        _img_shape = img.shape[:2]
        ret, corners = cv2.findChessboardCorners(
            img,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(img, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)

    N_OK = len(objpoints)
    if N_OK == 0:
        print("No valid chessboards found - skipping calibration.")
        return None, None, None  # or raise a custom error

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        _img_shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    )
    print(f"Found {N_OK} valid images for calibration (RMS={rms})")
    print("DIM =", _img_shape[::-1])
    print("K =", K)
    print("D =", D)

    return K, D, _img_shape[::-1]


def calibrate_robust(imgs, checkerboard=(8, 6), square_size=1.0, min_views=8,
              max_prune_frac=0.3, use_check_cond=False):
    """
    Robust fisheye calibration.
    - checkerboard: (cols, rows) inner corners
    - square_size: physical size of one square (any consistent unit)
    - min_views: minimum valid views to keep
    - max_prune_frac: up to this fraction of worst views may be dropped
    - use_check_cond: if True, keep CALIB_CHECK_COND; otherwise avoid hard fail
    """
    # --- config
    sb_flags = (
        cv2.CALIB_CB_EXHAUSTIVE |
        cv2.CALIB_CB_ACCURACY |
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    classic_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE |
        cv2.CALIB_CB_FAST_CHECK
    )
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)

    calib_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
        cv2.fisheye.CALIB_FIX_SKEW
    )
    if use_check_cond:
        calib_flags |= cv2.fisheye.CALIB_CHECK_COND  # can cause hard failure

    # --- prepare object points for one view (scale by square_size)
    cols, rows = checkerboard
    objp = np.zeros((1, cols * rows, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints, imgpoints, shapes, kept_idx = [], [], [], []

    # --- detect corners
    for i, img in enumerate(imgs):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        shape = gray.shape[:2]
        ret = False
        # 1) SB (more robust)
        try:
            ret, corners = cv2.findChessboardCornersSB(gray, checkerboard, sb_flags)
        except Exception:
            ret = False
        # 2) fallback
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, checkerboard, classic_flags)

        if not ret:
            continue

        # Subpixel refine
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)

        # Filter weak detections: ensure decent size / spread
        xs = corners[:, 0, 0]; ys = corners[:, 0, 1]
        bb_w = xs.max() - xs.min()
        bb_h = ys.max() - ys.min()
        area_frac = (bb_w * bb_h) / (shape[1] * shape[0])
        # Require the board to occupy at least 2% of the image and not be degenerate
        if area_frac < 0.02 or bb_w < 20 or bb_h < 20:
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        shapes.append(shape)
        kept_idx.append(i)

    N = len(objpoints)
    if N < min_views:
        print(f"Only {N} valid views (need ≥ {min_views}). Aborting.")
        return None, None, None

    # Sanity: consistent image size
    H, W = shapes[0]
    if any(s != (H, W) for s in shapes):
        print("Mixed image sizes detected; please resize all images to the same dimensions.")
        return None, None, None

    # --- helper: run calibration and compute per-view errors
    def run_once(op, ip):
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), np.float64) for _ in range(len(op))]
        tvecs = [np.zeros((1, 1, 3), np.float64) for _ in range(len(op))]
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            op, ip, (W, H), K, D, rvecs, tvecs,
            calib_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7),
        )
        # Per-view reprojection error
        per_view_err = []
        for j in range(len(op)):
            proj, _ = cv2.fisheye.projectPoints(op[j], rvecs[j], tvecs[j], K, D)
            err = np.linalg.norm(ip[j] - proj) / proj.shape[0]
            per_view_err.append(float(err))
        return rms, K, D, rvecs, tvecs, np.array(per_view_err)

    # --- robust loop: prune worst views if ill-conditioned / high error
    op = objpoints[:]
    ip = imgpoints[:]
    max_drop = max(1, int(max_prune_frac * len(op)))
    dropped = 0

    while True:
        try:
            rms, K, D, rvecs, tvecs, per_err = run_once(op, ip)
        except cv2.error as e:
            # If it’s the ill-conditioned failure, try dropping the last-added view
            if "CALIB_CHECK_COND" in str(e) or "Ill-conditioned" in str(e):
                if not use_check_cond:
                    # If CHECK_COND is off, we shouldn't be here; break to avoid loop
                    print("Ill-conditioned. Consider taking more diverse images.")
                    return None, None, None
                if len(op) > min_views and dropped < max_drop:
                    # Drop the most recent view and retry
                    op.pop(); ip.pop()
                    dropped += 1
                    continue
                print("Failed due to ill-conditioned data. Collect more/varied views.")
                return None, None, None
            else:
                raise

        # If success, optionally prune extreme outliers and refit once
        if len(op) > min_views and dropped < max_drop:
            worst_idx = int(per_err.argmax())
            if per_err[worst_idx] > (2.5 * np.median(per_err)):  # heuristic threshold
                op.pop(worst_idx); ip.pop(worst_idx)
                dropped += 1
                continue

        # done
        print(f"Calibrated with {len(op)} views (dropped {dropped}), RMS={rms:.4f}")
        print("DIM =", (W, H))
        print("K =", K)
        print("D =", D.T)
        return K, D, (W, H)