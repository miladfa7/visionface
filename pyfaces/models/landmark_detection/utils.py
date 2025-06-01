


def medipipe_mesh_landmark_names():
    FACE_MESH_LANDMARK_NAMES = {
        # Face contour (jawline and outer face boundary)
        0: "face_contour_0", 1: "nose_tip", 2: "nose_bridge_2", 3: "nose_bridge_3", 4: "nose_bridge_4",
        5: "nose_bridge_5", 6: "nose_bridge_6", 7: "left_eye_inner_7", 8: "nose_bridge_8", 9: "forehead_center_9",
        10: "forehead_center_10", 11: "face_contour_11", 12: "face_contour_12", 13: "upper_lip_center_13", 
        14: "lower_lip_center_14", 15: "face_contour_15", 16: "face_contour_16", 17: "upper_lip_17",
        18: "chin_18", 19: "face_contour_19", 20: "face_contour_20",
        
        # Left eye region (from viewer's perspective)
        33: "left_eye_outer_33", 7: "left_eye_inner_7", 163: "left_eye_lower_163", 144: "left_eye_lower_144",
        145: "left_eye_lower_145", 153: "left_eye_lower_153", 154: "left_eye_lower_154", 155: "left_eye_lower_155",
        133: "left_eye_outer_133", 173: "left_eye_upper_173", 157: "left_eye_upper_157", 158: "left_eye_upper_158",
        159: "left_eye_upper_159", 160: "left_eye_upper_160", 161: "left_eye_upper_161", 246: "left_eye_lower_246",
        
        # Right eye region
        362: "right_eye_inner_362", 382: "right_eye_upper_382", 381: "right_eye_upper_381", 380: "right_eye_upper_380",
        374: "right_eye_upper_374", 373: "right_eye_upper_373", 390: "right_eye_upper_390", 249: "right_eye_outer_249",
        263: "right_eye_outer_263", 466: "right_eye_lower_466", 388: "right_eye_lower_388", 387: "right_eye_lower_387",
        386: "right_eye_lower_386", 385: "right_eye_lower_385", 384: "right_eye_lower_384", 398: "right_eye_upper_398",
        
        # Left eyebrow
        46: "left_eyebrow_inner_46", 53: "left_eyebrow_53", 52: "left_eyebrow_52", 51: "left_eyebrow_51",
        48: "left_eyebrow_48", 115: "left_eyebrow_115", 131: "left_eyebrow_outer_131", 134: "left_eyebrow_134",
        102: "left_eyebrow_102", 49: "left_eyebrow_49", 220: "left_eyebrow_220", 305: "left_eyebrow_305",
        
        # Right eyebrow
        276: "right_eyebrow_inner_276", 283: "right_eyebrow_283", 282: "right_eyebrow_282", 295: "right_eyebrow_295",
        285: "right_eyebrow_285", 336: "right_eyebrow_336", 296: "right_eyebrow_296", 334: "right_eyebrow_334",
        293: "right_eyebrow_293", 300: "right_eyebrow_300", 441: "right_eyebrow_outer_441",
        
        # Nose detailed points
        168: "nose_bridge_168", 195: "nostril_left_195", 197: "nostril_left_197", 196: "nostril_left_196",
        3: "nose_bridge_3", 51: "nose_left_51", 48: "nose_left_48", 115: "nose_left_115", 131: "nose_left_131",
        134: "nose_left_134", 102: "nose_left_102", 49: "nose_left_49", 220: "nose_left_220", 305: "nose_left_305",
        278: "nose_right_278", 279: "nose_right_279", 420: "nostril_right_420", 456: "nostril_right_456",
        248: "nose_right_248", 281: "nose_right_281", 275: "nose_right_275",
        
        # Lips outer boundary
        61: "mouth_left_corner_61", 84: "upper_lip_left_84", 17: "upper_lip_17", 314: "upper_lip_right_314",
        405: "mouth_right_corner_405", 320: "lower_lip_right_320", 307: "lower_lip_307", 375: "lower_lip_375",
        321: "lower_lip_321", 308: "lower_lip_308", 324: "lower_lip_324", 318: "lower_lip_318",
        
        # Lips inner boundary
        78: "inner_lip_upper_78", 95: "inner_lip_upper_95", 88: "inner_lip_upper_88", 178: "inner_lip_upper_178",
        87: "inner_lip_upper_87", 14: "inner_lip_lower_14", 317: "inner_lip_lower_317", 402: "inner_lip_lower_402",
        318: "inner_lip_lower_318", 324: "inner_lip_lower_324", 308: "inner_lip_lower_308", 415: "inner_lip_lower_415",
        
        # Additional mouth points
        291: "mouth_right_corner_291", 303: "mouth_upper_303", 267: "mouth_lower_267", 269: "mouth_lower_269",
        270: "mouth_lower_270", 267: "mouth_lower_267", 271: "mouth_lower_271", 272: "mouth_lower_272",
        
        # Chin and jaw
        175: "chin_left_175", 199: "chin_bottom_199", 175: "chin_right_175", 18: "chin_center_18",
        175: "jaw_left_175", 199: "jaw_bottom_199", 175: "jaw_right_175",
        
        # Cheek regions
        116: "left_cheek_116", 117: "left_cheek_117", 118: "left_cheek_118", 119: "left_cheek_119",
        120: "left_cheek_120", 121: "left_cheek_121", 126: "left_cheek_126", 142: "left_cheek_142",
        36: "left_cheek_36", 205: "left_cheek_205", 206: "left_cheek_206", 207: "left_cheek_207",
        213: "left_cheek_213", 192: "left_cheek_192", 147: "left_cheek_147",
        
        345: "right_cheek_345", 346: "right_cheek_346", 347: "right_cheek_347", 348: "right_cheek_348",
        349: "right_cheek_349", 350: "right_cheek_350", 451: "right_cheek_451", 452: "right_cheek_452",
        453: "right_cheek_453", 464: "right_cheek_464", 435: "right_cheek_435", 410: "right_cheek_410",
        454: "right_cheek_454",
        
        # Forehead points
        151: "forehead_151", 337: "forehead_337", 299: "forehead_299", 333: "forehead_333",
        298: "forehead_298", 301: "forehead_301", 284: "forehead_284", 251: "forehead_251",
        389: "forehead_389", 356: "forehead_356", 454: "forehead_454", 323: "forehead_323",
        361: "forehead_361", 340: "forehead_340",
        
        # Temple regions
        103: "left_temple_103", 67: "left_temple_67", 109: "left_temple_109", 338: "temple_338",
        332: "right_temple_332", 297: "right_temple_297",
    }

    # Fill remaining indices with generic names
    for i in range(478):
        if i not in FACE_MESH_LANDMARK_NAMES:
            # Determine general region based on index ranges
            if i < 17:
                FACE_MESH_LANDMARK_NAMES[i] = f"face_contour_{i}"
            elif 17 <= i < 68:
                FACE_MESH_LANDMARK_NAMES[i] = f"right_eyebrow_region_{i}"
            elif 68 <= i < 103:
                FACE_MESH_LANDMARK_NAMES[i] = f"nose_bridge_region_{i}"
            elif 103 <= i < 134:
                FACE_MESH_LANDMARK_NAMES[i] = f"right_eye_region_{i}"
            elif 134 <= i < 155:
                FACE_MESH_LANDMARK_NAMES[i] = f"left_eye_region_{i}"
            elif 155 <= i < 180:
                FACE_MESH_LANDMARK_NAMES[i] = f"left_eyebrow_region_{i}"
            elif 180 <= i < 200:
                FACE_MESH_LANDMARK_NAMES[i] = f"nose_tip_region_{i}"
            elif 200 <= i < 220:
                FACE_MESH_LANDMARK_NAMES[i] = f"nostril_region_{i}"
            elif 220 <= i < 250:
                FACE_MESH_LANDMARK_NAMES[i] = f"cheek_region_{i}"
            elif 250 <= i < 300:
                FACE_MESH_LANDMARK_NAMES[i] = f"mouth_region_{i}"
            elif 300 <= i < 340:
                FACE_MESH_LANDMARK_NAMES[i] = f"chin_jaw_region_{i}"
            elif 340 <= i < 400:
                FACE_MESH_LANDMARK_NAMES[i] = f"right_face_region_{i}"
            else:
                FACE_MESH_LANDMARK_NAMES[i] = f"face_mesh_{i}"

    return FACE_MESH_LANDMARK_NAMES