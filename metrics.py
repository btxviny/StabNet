import cv2
import numpy as np
import sys

def metric(original_video_path, pred_video_path,shape=(128,128)):
    ''' Inputs: 
            path1: path to ground truth stable video
            path2: path to generated stable video
        Outputs:
            cropping_score
            distortion_score
            pixel_score
            stability_score
    '''
    # Create brute-force matcher object
    sys.stdout.flush()
    bf = cv2.BFMatcher()
    sift = cv2.SIFT_create()

    # Apply the homography transformation if we have enough good matches
    MIN_MATCH_COUNT = 10
    ratio = 0.7
    thresh = 5.0

    CR_seq = []
    DV_seq = []
    Pt = np.eye(3)
    P_seq = []
    pixel_loss = []

    # Video loading
    H,W = shape
    #load both videos
    cap1 = cv2.VideoCapture(original_video_path)
    cap2 = cv2.VideoCapture(pred_video_path)
    frame_count = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)),int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    original_frames = np.zeros((frame_count,H,W,3),np.uint8)
    pred_frames = np.zeros_like(original_frames)
    for i in range(frame_count):
        ret1,img1 = cap1.read()
        ret2,img2 = cap2.read()
        if not ret1 or not ret2:
            break
        img1 = cv2.resize(img1,(W,H))
        img2 = cv2.resize(img2,(W,H))
        original_frames[i,...] = img1
        pred_frames[i,...] = img2

    for i in range(frame_count):
        img1 = original_frames[i,...]
        img1o = pred_frames[i,...]

        # Convert frames to grayscale
        a = (img1 / 255.0).astype(np.float32)
        b = (img1o/ 255.0).astype(np.float32)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1o = cv2.cvtColor(img1o, cv2.COLOR_BGR2GRAY)
        pixel_loss.append(np.mean((a-b)**2))
        # Detect the SIFT key points and compute the descriptors for the two images
        keyPoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keyPoints1o, descriptors1o = sift.detectAndCompute(img1o, None)

        # Match the descriptors
        matches = bf.knnMatch(descriptors1, descriptors1o, k=2)

        # Select the good matches using the ratio test
        goodMatches = []

        for m, n in matches:
            if m.distance < ratio * n.distance:
                goodMatches.append(m)

        if len(goodMatches) > MIN_MATCH_COUNT:
            # Get the good key points positions
            sourcePoints = np.float32([keyPoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            destinationPoints = np.float32([keyPoints1o[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

            # Obtain the homography matrix
            M, _ = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)

            # Obtain Scale, Translation, Rotation, Distortion value
            scaleRecovered = np.sqrt(M[0, 1] ** 2 + M[0, 0] ** 2)

            eigenvalues = np.abs(np.linalg.eigvals(M[0:2, 0:2]))
            eigenvalues = sorted(eigenvalues,reverse= True)
            DV = (eigenvalues[1] / eigenvalues[0]).astype(np.float32)

            CR_seq.append(1 / scaleRecovered)
            DV_seq.append(DV)

        # For Stability score calculation
        if i + 1 < frame_count:
            img2o = pred_frames[i+1,...]
            # Convert frame to grayscale
            img2o = cv2.cvtColor(img2o, cv2.COLOR_BGR2GRAY)

            keyPoints2o, descriptors2o = sift.detectAndCompute(img2o, None)
            matches = bf.knnMatch(descriptors1o, descriptors2o, k=2)
            goodMatches = []

            for m, n in matches:
                if m.distance < ratio * n.distance:
                    goodMatches.append(m)

            if len(goodMatches) > MIN_MATCH_COUNT:
                # Get the good key points positions
                sourcePoints = np.float32([keyPoints1o[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
                destinationPoints = np.float32([keyPoints2o[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

                # Obtain the homography matrix
                M, _ = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)

                P_seq.append(np.matmul(Pt, M))
                Pt = np.matmul(Pt, M)
        
        sys.stdout.write('\rFrame: ' + str(i) + '/' + str(frame_count))

    cap1.release()
    cap2.release()

    # Make 1D temporal signals
    P_seq_t = []
    P_seq_r = []

    for Mp in P_seq:
        transRecovered = np.sqrt(Mp[0, 2] ** 2 + Mp[1, 2] ** 2)
        thetaRecovered = np.arctan2(Mp[1, 0], Mp[0, 0]) * 180 / np.pi
        P_seq_t.append(transRecovered)
        P_seq_r.append(thetaRecovered)

    # FFT
    fft_t = np.fft.fft(P_seq_t)
    fft_r = np.fft.fft(P_seq_r)
    fft_t = np.abs(fft_t) ** 2
    fft_r = np.abs(fft_r) ** 2

    fft_t = np.delete(fft_t, 0)
    fft_r = np.delete(fft_r, 0)
    fft_t = fft_t[:len(fft_t) // 2]
    fft_r = fft_r[:len(fft_r) // 2]

    SS_t = np.sum(fft_t[:5]) / np.sum(fft_t)
    SS_r = np.sum(fft_r[:5]) / np.sum(fft_r)

    cropping_score = np.min([np.mean(CR_seq), 1])
    distortion_score = np.min(DV_seq)
    stability_score = (SS_t+SS_r)/2
    pixel_score = 1 - np.mean(pixel_loss)
    out = f'\ncropping score:{cropping_score:.3f}\tdistortion score:{distortion_score:.3f}\tstability:{stability_score:.3f}\tpixel:{pixel_score:.3f}\n'
    sys.stdout.write(out)
    return cropping_score, distortion_score, stability_score, pixel_score

