import concurrent.futures
from deepface import DeepFace


def process(requestModel):
    print("verify faces...")
    return verify_faces(requestModel.photoModel, requestModel.photoFilter, requestModel.distance)


def verify_image(model_image, image_to_filter):
    try:
        result = DeepFace.verify(img1_path=model_image,
                                 img2_path=image_to_filter,
                                 detector_backend="mtcnn",
                                 distance_metric="cosine",
                                 model_name="ArcFace"
                                 )
        distance = result["distance"]
        filtered_image = image_to_filter.split(",")[1]
        return {"distance": distance, "filtered_image": filtered_image}
    except Exception as e:
        print(f"Error processing image: {str(e)}")
    return {"distance": None, "filtered_image": None}


def verify_faces(model_images, images_to_filter, distance_threshold):
    verified_photos = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for image_to_filter in images_to_filter:
            image_filter_base_64 = f'data:image/jpeg;base64,{image_to_filter}'
            for model_image in model_images:
                future = executor.submit(verify_image, model_image, image_filter_base_64)
                futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            distance = results["distance"]
            print(f'>>> {distance}')
            if distance is not None and distance < distance_threshold:
                verified_photos.add(results["filtered_image"])
    return list(verified_photos)


def verify_faces_sync(model_images, images_to_filter, distance_threshold):
    verified_photos = set()
    index = 0
    for image_to_filter in images_to_filter:
        index += 1
        image_filter_base_64 = f'data:image/jpeg;base64,{image_to_filter}'
        for model_image in model_images:
            try:
                result = DeepFace.verify(img1_path=model_image,
                                         img2_path=image_filter_base_64,
                                         detector_backend="mtcnn",
                                         distance_metric="cosine",
                                         model_name="ArcFace"
                                         )
                print(f'>>> {index}: {result["distance"]}')
                if result["distance"] < distance_threshold:
                    verified_photos.add(image_filter_base_64.split(",")[1])
            except Exception as e:
                print(f"Error processing image: {str(e)}")
    return list(verified_photos)
