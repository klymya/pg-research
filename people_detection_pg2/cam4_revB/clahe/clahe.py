import numpy as np

def apply_clahe_channel(channel, clip_limit=3.0, tile_grid_size=(8, 8), nbins=256):
    h, w = channel.shape
    tile_h = h // tile_grid_size[0]
    tile_w = w // tile_grid_size[1]
    
    histograms = np.zeros((tile_grid_size[0], tile_grid_size[1], nbins), dtype=np.float32)

    # Step 1: Compute histograms for each tile
    for i in range(tile_grid_size[0]):
        for j in range(tile_grid_size[1]):
            y0 = i * tile_h
            x0 = j * tile_w
            tile = channel[y0:y0+tile_h, x0:x0+tile_w]

            hist, _ = np.histogram(tile, bins=nbins, range=(0, 256))

            # Step 2: Apply OpenCV-style clip limit
            clip_limit_pixels = clip_limit * tile.size / nbins
            excess = hist - clip_limit_pixels
            excess[excess < 0] = 0
            total_excess = int(np.sum(excess))
            hist = np.clip(hist, 0, clip_limit_pixels)

            # Redistribute excess uniformly
            hist += total_excess // nbins
            remainder = total_excess % nbins
            if remainder > 0:
                hist[:remainder] += 1

            # Normalize to PDF
            histograms[i, j] = hist / np.sum(hist)

    # Step 3: Convert PDFs to CDFs
    cdfs = np.cumsum(histograms, axis=-1)

    # Step 4: Interpolate pixel values from CDFs
    output = np.zeros_like(channel)
    for y in range(h):
        for x in range(w):
            i = min(y // tile_h, tile_grid_size[0] - 1)
            j = min(x // tile_w, tile_grid_size[1] - 1)

            # Relative position inside tile
            y1 = i * tile_h
            x1 = j * tile_w
            dy = (y - y1) / tile_h
            dx = (x - x1) / tile_w

            val = channel[y, x]

            def get_cdf(ii, jj):
                ii = np.clip(ii, 0, tile_grid_size[0] - 1)
                jj = np.clip(jj, 0, tile_grid_size[1] - 1)
                return cdfs[ii, jj, val]

            # Bilinear interpolation between 4 tile CDFs
            c00 = get_cdf(i, j)
            c01 = get_cdf(i, j + 1)
            c10 = get_cdf(i + 1, j)
            c11 = get_cdf(i + 1, j + 1)

            interp = (
                c00 * (1 - dx) * (1 - dy) +
                c01 * dx * (1 - dy) +
                c10 * (1 - dx) * dy +
                c11 * dx * dy
            )

            output[y, x] = np.clip(interp * 255, 0, 255)

    return output.astype(np.uint8)
