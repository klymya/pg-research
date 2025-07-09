def apply_clahe_channel(channel, clip_limit=3.0, tile_grid_size=(8, 8), nbins=256):
    height = len(channel)
    width = len(channel[0])
    tile_h = height // tile_grid_size[0]
    tile_w = width // tile_grid_size[1]

    # Step 1: compute histograms and CDFs for each tile
    cdfs = []
    for i in range(tile_grid_size[0]):
        row_cdfs = []
        for j in range(tile_grid_size[1]):
            y0 = i * tile_h
            x0 = j * tile_w
            hist = [0] * nbins

            # Build histogram
            for y in range(y0, y0 + tile_h):
                for x in range(x0, x0 + tile_w):
                    val = channel[y][x]
                    hist[val] += 1

            # Clip histogram using OpenCV-style formula
            tile_size = tile_h * tile_w
            clip_limit_px = clip_limit * tile_size / nbins
            excess = 0
            for k in range(nbins):
                if hist[k] > clip_limit_px:
                    excess += hist[k] - clip_limit_px
                    hist[k] = int(clip_limit_px)

            # Redistribute excess uniformly
            step = excess // nbins
            remainder = int(excess % nbins)
            for k in range(nbins):
                hist[k] += step
            for k in range(remainder):
                hist[k] += 1

            # Compute normalized CDF
            cdf = [0] * nbins
            acc = 0
            total = sum(hist)
            for k in range(nbins):
                acc += hist[k]
                cdf[k] = acc / total  # normalized
            row_cdfs.append(cdf)
        cdfs.append(row_cdfs)

    # Step 2: Interpolate and map pixels
    output = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            val = channel[y][x]
            i = min(y // tile_h, tile_grid_size[0] - 1)
            j = min(x // tile_w, tile_grid_size[1] - 1)

            # Tile origins
            y1 = i * tile_h
            x1 = j * tile_w
            dy = (y - y1) / tile_h
            dx = (x - x1) / tile_w

            def get_cdf(ii, jj):
                ii = max(0, min(ii, tile_grid_size[0] - 1))
                jj = max(0, min(jj, tile_grid_size[1] - 1))
                return cdfs[ii][jj][val]

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

            output[y][x] = min(255, max(0, int(round(interp * 255))))

    return output
