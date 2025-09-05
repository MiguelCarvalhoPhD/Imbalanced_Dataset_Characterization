
import cupy as cp
import numpy as np
from cuml.svm import SVC, LinearSVC
from cuml.metrics import confusion_matrix, pairwise_distances
from cuml.neighbors import NearestNeighbors
#from imblearn.over_sampling import SMOTE

def compute_F2_imbalanced_gpu(data, labels):
    """
    Compute the F2 modified metric for a dataset on GPU without using a for loop.
    
    Parameters:
    - data: A 2D numpy array where each row is a sample and each column is a feature.
    - labels: A 1D numpy array of class labels (0 or 1) for each sample in data.
    
    Returns:
    - The F2 metric for the dataset.
    """
    
    #Convert data and labels to CuPy arrays
    data_gpu = cp.array(data)
    labels_gpu = cp.array(labels)

    #avoid 0 values in binary atributes
    non_binary_labels = cp.array([len(cp.unique(b)) > 3  for b in data_gpu.T])

    if cp.any(non_binary_labels):

        data_gpu = data_gpu[:,non_binary_labels]

        #Split the data into two classes based on labels
        class_1_data = data_gpu[labels_gpu == 0]
        class_2_data = data_gpu[labels_gpu == 1]
        
        #Compute min and max for each feature for both classes
        max_c1 = cp.max(class_1_data, axis=0)
        max_c2 = cp.max(class_2_data, axis=0)
        
        min_c1 = cp.min(class_1_data, axis=0)
        min_c2 = cp.min(class_2_data, axis=0)
        
        #ompute overlap for all features
        minmax = cp.minimum(max_c1, max_c2)
        maxmin = cp.maximum(min_c1, min_c2)
        overlap = cp.maximum(0, minmax - maxmin)

        #avoid min_c1 = max_c1
        range_c1 = max_c1 - min_c1
        range_c2 = max_c2 - min_c2

        #Compute F2 values for all features
        f_c1 = overlap / range_c1
        f_c2 = overlap / range_c2
        
        #Compute the product of F2 values for all features
        F2_c1 = cp.prod(f_c1)
        F2_c2 = cp.prod(f_c2)
            
    else:

        F2_c1 = cp.NAN
        F2_c2 = cp.NaN
    
    return float(cp.asnumpy(F2_c1)), float(cp.asnumpy(F2_c2)), float(cp.mean(cp.array([F2_c1,F2_c2])).get())

def compute_F3_imbalanced_gpu(data, labels):
    """
    Compute the F3 metric for a dataset using GPU.
    
    Parameters:
    - data: A 2D numpy array where each row is a sample and each column is a feature.
    - labels: A 1D numpy array of class labels (0 or 1) for each sample in data.
    
    Returns:
    - The F3 metric for the dataset.
    """
    
    # Transfer data to GPU
    data_gpu = cp.array(data)
    labels_gpu = cp.array(labels)
    
    # Split the data into two classes based on labels
    class_1_data = data_gpu[labels_gpu == 0]
    class_2_data = data_gpu[labels_gpu == 1]
    
    # Compute min and max for each feature for both classes
    max_c1 = cp.max(class_1_data, axis=0)
    max_c2 = cp.max(class_2_data, axis=0)
    
    min_c1 = cp.min(class_1_data, axis=0)
    min_c2 = cp.min(class_2_data, axis=0)
    
    # Compute overlap regions for all features
    minmax = cp.minimum(max_c1, max_c2)
    maxmin = cp.maximum(min_c1, min_c2)
    
    # Compute overlap counts for all features
    overlap_count_c1 = cp.sum((class_1_data >= maxmin) & (class_1_data <= minmax), axis=0)
    overlap_count_c2 = cp.sum((class_2_data >= maxmin) & (class_2_data <= minmax), axis=0)
    
    # Compute the overlap ratio for all features
    overlap_ratio_c1 = overlap_count_c1 / class_1_data.shape[0]
    overlap_ratio_c2 = overlap_count_c2 / class_2_data.shape[0]

    #avoid 0 values in binary atributes
    non_binary_labels = cp.array([len(cp.unique(b)) > 3  for b in data_gpu.T])

    # Find the minimum overlap ratios
    if cp.any(non_binary_labels):
        min_overlap_ratio_c1 = cp.min(overlap_ratio_c1[non_binary_labels])
        min_overlap_ratio_c2 = cp.min(overlap_ratio_c2[non_binary_labels])
    else:
        min_overlap_ratio_c1 = cp.nan
        min_overlap_ratio_c2 = cp.nan
    
    # Transfer results back to CPU
    min_overlap_ratio_c1 = cp.asnumpy(min_overlap_ratio_c1)
    min_overlap_ratio_c2 = cp.asnumpy(min_overlap_ratio_c2)
    
    return min_overlap_ratio_c1, min_overlap_ratio_c2, (min_overlap_ratio_c1+min_overlap_ratio_c2)/2

def compute_F4_imbalanced_gpu(data,labels):

    # Transfer data to GPU
    data_gpu = cp.array(data)
    labels_gpu = cp.array(labels)

    #eliminate binary features
    non_binary_labels = cp.array([len(cp.unique(b)) > 2  for b in data_gpu.T])

    # Find the minimum overlap ratios
    if cp.any(non_binary_labels):

        data_gpu = data_gpu[:,non_binary_labels]

        # Split the data into two classes based on labels
        class_1_data = data_gpu[labels_gpu == 0]
        class_2_data = data_gpu[labels_gpu == 1]

        # Compute min and max for each feature for both classes
        max_c1 = cp.max(class_1_data, axis=0)
        max_c2 = cp.max(class_2_data, axis=0)
        
        min_c1 = cp.min(class_1_data, axis=0)
        min_c2 = cp.min(class_2_data, axis=0)
        
        # Compute overlap regions for all features
        minmax = cp.minimum(max_c1, max_c2)
        maxmin = cp.maximum(min_c1, min_c2)
        
        # Compute overlap counts for all features
        overlap_matrix = (data_gpu > minmax) | (data_gpu < maxmin)
        
        # compute most discriminate features
        overlap_counts = cp.sum(overlap_matrix,axis=0)

        #sort matrix based on most discriminate features
        overlap_matrix = overlap_matrix[:,cp.flip(cp.argsort(overlap_counts))]

        #variable to analyze correctly classified samples
        classified_samples = cp.zeros(data_gpu.shape[0],dtype=bool)

        #iterate over all features
        for i in range(data_gpu.shape[1]):
            classified_samples = cp.logical_or(classified_samples,overlap_matrix[:,i])

            #check if all samples are correctly classified
            if cp.sum(classified_samples) == data_gpu.shape[0]:
                print('done')
                break

        #compute metric for both classes and mean
        F4_c0 = cp.sum(cp.logical_and(labels_gpu==0,~classified_samples)) / cp.sum(labels_gpu==0)
        F4_c1 = cp.sum(cp.logical_and(labels_gpu==1,~classified_samples)) / cp.sum(labels_gpu==1)

    else:

        F4_c0 = cp.nan
        F4_c1 = cp.nan
    
    return float(F4_c0), float(F4_c1), float((F4_c0+F4_c1)/2)

def compute_L1_imbalanced_gpu(data, labels):
    """
    Compute the L1 metric for a dataset.
    
    Parameters:
    - data: A 2D numpy array where each row is a sample and each column is a feature.
    - labels: A 1D numpy array of class labels for each sample in data.
    
    Returns:
    - The L1 metric for the dataset.
    """
    
    # Convert data and labels to CuPy arrays
    data_gpu = cp.array(data).astype(cp.float32)
    labels_gpu = cp.array(labels).astype(cp.int32)

    #define c1 and c0 labels
    labels_c1 = labels_gpu == 1
    labels_c0 = labels_gpu == 0

    # Train a linear classifier
    clf = LinearSVC()
    clf.fit(data_gpu, labels_gpu)
    
    # Predict the labels
    predictions = clf.predict(data_gpu).astype(cp.int32)
    incorrect = predictions != labels_gpu
    
    # Compute the signed distance of each sample to the decision boundary
    L1_c1 = cp.sum(cp.abs(clf.decision_function(data_gpu[labels_c1])[incorrect[labels_c1]])) / cp.sum(labels_c1)
    L1_c0 = cp.sum(cp.abs(clf.decision_function(data_gpu[labels_c0])[incorrect[labels_c0]])) / cp.sum(labels_c0)
    L1 = (L1_c1+L1_c0)/2
    
    return float(L1_c0), float(L1_c1), float(1/(1+L1))

def compute_L2_imbalanced_gpu(data, labels):
    """
    Compute the degree of non-linearity for a dataset using the error of a linear classifier.
    
    Parameters:
    - data: A 2D numpy array where each row is a sample and each column is a feature.
    - labels: A 1D numpy array of class labels for each sample in data.
    
    Returns:
    - Error rates for each class.

    """

    # Convert data and labels to CuPy arrays
    data_gpu = cp.array(data).astype(cp.float32)
    labels_gpu = cp.array(labels).astype(cp.int32)

    # Train a linear classifier
    clf = LinearSVC()
    clf.fit(data_gpu, labels_gpu)

    #define c1 and c0 labels
    labels_c1 = labels_gpu == 1
    labels_c0 = labels_gpu == 0
 
    # Predict the labels
    predictions = clf.predict(data_gpu)
    
    # Compute the error rate for each class
    error_rate_c0 = cp.mean((predictions[labels_c0] != labels_gpu[labels_c0]))
    error_rate_c1 = cp.mean((predictions[labels_c1] != labels_gpu[labels_c1]))
    
    return float(error_rate_c0), float(error_rate_c1), float((error_rate_c0+error_rate_c1)/2)

def compute_L3_imbalanced_gpu(data, labels):
    """
    Compute the degree of non-linearity for a dataset using the error of a linear classifier.
    
    Parameters:
    - data: A 2D numpy array where each row is a sample and each column is a feature.
    - labels: A 1D numpy array of class labels for each sample in data.
    
    Returns:
    - Error rates for each class.

    """

    # Convert data and labels to CuPy arrays
    data_gpu = cp.array(data).astype(cp.float32)
    labels_gpu = cp.array(labels).astype(cp.int32)

    # Train a linear classifier
    clf = LinearSVC()
    clf.fit(data_gpu, labels_gpu)

    #define synthetic variables
    test_data_gpu   = cp.zeros(data_gpu.shape,dtype=cp.float32)
    test_labels_gpu = cp.zeros(len(labels_gpu),dtype=cp.int32)

    ### generate synthetic data
    for label in range(2):

        # Find indices of the current class
        class_indices = cp.where(labels_gpu == label)[0]
        
        # If there are not enough samples to perform interpolation, skip the class
        if len(class_indices) < 2:
            continue
        
        # Determine the number of synthetic samples for the current class
        num_synthetic_samples_class = len(class_indices)


        # Randomly select two sets of samples from the current class for interpolation
        sample_indices_1 = cp.random.choice(class_indices, num_synthetic_samples_class, replace=True)
        sample_indices_2 = cp.random.choice(class_indices, num_synthetic_samples_class, replace=True)
        
        # Generate all alpha values at once
        alphas = cp.random.rand(num_synthetic_samples_class)
        
        # Compute synthetic samples
        synthetic_samples = (alphas[:, None] * data_gpu[sample_indices_1] + (1 - alphas[:, None]) * data_gpu[sample_indices_2])
        
        # Assign synthetic samples and labels to the test data and labels arrays
        start_index = 0 if label == 0 else -num_synthetic_samples_class
        end_index = num_synthetic_samples_class if label == 0 else None
        test_data_gpu[start_index:end_index] = synthetic_samples
        test_labels_gpu[start_index:end_index] = label

    #define c1 and c0 labels
    labels_c1 = test_labels_gpu == 1
    labels_c0 = test_labels_gpu == 0
 
    # Predict the labels
    predictions = clf.predict(test_data_gpu)
    
    # Compute the error rate for each class
    error_rate_c0 = cp.mean((predictions[labels_c0] != test_labels_gpu[labels_c0]))
    error_rate_c1 = cp.mean((predictions[labels_c1] != test_labels_gpu[labels_c1]))
    
    return float(error_rate_c0), float(error_rate_c1), float((error_rate_c0+error_rate_c1)/2)

def compute_N3_imbalanced_gpu(data,labels):

    # Transfer data and labels to GPU
    data_gpu = cp.array(data).astype(cp.float32)
    labels_gpu = cp.array(labels).astype(cp.int16)

    knn = NearestNeighbors(n_neighbors=2) # Find the single closest neighbor
    knn.fit(data_gpu)  # Train the model on data

    _, indexes = knn.kneighbors(data_gpu,2)
    indexes    = indexes[:,-1]

    # Check if the label of each instance is different from the label of its nearest neighbor
    errors = labels_gpu != labels_gpu[indexes]

    # check labels indexes for majority and minority class
    c0 = labels_gpu == 0
    c1 = labels_gpu == 1

    errors_c0 = cp.sum(errors[c0])
    errors_c1 = cp.sum(errors[c1])

    # Compute N3
    N3_c0 = errors_c0 / cp.sum(c0)
    N3_c1 = errors_c1 / cp.sum(c1)
    N3    = (N3_c0+N3_c1)/2

    return float(N3_c0), float(N3_c1), float(N3)

def compute_N2_imbalanced_gpu(data,labels):

    # Transfer data and labels to GPU
    data_gpu = cp.array(data).astype(cp.float32)
    labels_gpu = cp.array(labels).astype(cp.int16)

    #separate data
    data_gpu_c0 = data_gpu[labels_gpu==0,:]
    data_gpu_c1 = data_gpu[labels_gpu==1,:]

    #avoid not having aneough nearest ceighbours
    if (data_gpu_c0.shape[0] < 2) or (data_gpu_c1.shape[0] < 2):

        return np.NaN, np.NaN, np.NaN
    
    else:

        #compute N2 for c0
        knn = NearestNeighbors(n_neighbors=2) # Find the single closest neighbor
        knn.fit(data_gpu_c0)  # Train the model on data

        closest_distances_c0 ,_ = knn.kneighbors(data_gpu_c0,2)
        closest_distances_c0    = closest_distances_c0[:,-1]

        furthers_distances_c1,_  = knn.kneighbors(data_gpu_c1,2)
        furthers_distances_c1    = furthers_distances_c1[:,0]

        #compute N2 for c1
        knn = NearestNeighbors(n_neighbors=2) # Find the single closest neighbor
        knn.fit(data_gpu_c1)  # Train the model on data

        closest_distances_c1 ,_ = knn.kneighbors(data_gpu_c1,2)
        closest_distances_c1    = closest_distances_c1[:,-1]

        furthers_distances_c0,_  = knn.kneighbors(data_gpu_c0,2)
        furthers_distances_c0    = furthers_distances_c0[:,0]

        N2_c0 = cp.sum(closest_distances_c0) / cp.sum(furthers_distances_c0)
        N2_c0 = (N2_c0)/(1+N2_c0)

        N2_c1 = cp.sum(closest_distances_c1) / cp.sum(furthers_distances_c1)
        N2_c1 = (N2_c1)/(1+N2_c1)

        #compute mean
        N2 = (N2_c1+N2_c0)/2

        #del closest_distances_c1, furthers_distances_c1, knn

        return float(N2_c0), float(N2_c1), float(N2)

def compute_N4_imbalanced_gpu(data,labels):

    # Convert data and labels to CuPy arrays
    data_gpu = cp.array(data)
    labels_gpu = cp.array(labels).astype(cp.int16)

    #define synthetic variables
    test_data_gpu   = cp.zeros(data_gpu.shape,dtype=cp.float32)
    test_labels_gpu = cp.zeros(len(labels_gpu),dtype=cp.int16)

    ### generate synthetic data
    for label in range(2):

        # Find indices of the current class
        class_indices = cp.where(labels_gpu == label)[0]
        
        # If there are not enough samples to perform interpolation, skip the class
        if len(class_indices) < 2:
            continue
        
        # Determine the number of synthetic samples for the current class
        num_synthetic_samples_class = len(class_indices)

        # Randomly select two sets of samples from the current class for interpolation
        sample_indices_1 = cp.random.choice(class_indices, num_synthetic_samples_class, replace=False)
        sample_indices_2 = cp.random.choice(class_indices, num_synthetic_samples_class, replace=False)
        
        # Generate all alpha values at once
        alphas = cp.random.rand(num_synthetic_samples_class)
        
        # Compute synthetic samples
        synthetic_samples = (alphas[:, None] * data_gpu[sample_indices_1] + (1 - alphas[:, None]) * data_gpu[sample_indices_2])
        
        # Assign synthetic samples and labels to the test data and labels arrays
        start_index = 0 if label == 0 else -num_synthetic_samples_class
        end_index = num_synthetic_samples_class if label == 0 else None
        test_data_gpu[start_index:end_index] = synthetic_samples
        test_labels_gpu[start_index:end_index] = label

    #compute nearest neighbours
    knn = NearestNeighbors(n_neighbors=2) # Find the single closest neighbor
    knn.fit(data_gpu)  # Train the model on data

    _, indexes = knn.kneighbors(test_data_gpu,1)
    indexes    = indexes[:,0] 

    #compute errors
    errors = labels_gpu[indexes] != test_labels_gpu

    #compute N4
    labels_c0 = test_labels_gpu == 0
    labels_c1 = test_labels_gpu == 1

    N4_c0 = cp.sum(errors[labels_c0]) / cp.sum(labels_c0)
    N4_c1 = cp.sum(errors[labels_c1]) / cp.sum(labels_c1)

    N4 = (N4_c0+N4_c1)/2

    return float(N4_c0), float(N4_c1), float(N4)

def T1_class_loop(data_gpu_c0,furthest_distance_index,furthest_distances_c0):

    #define matrix to store if data point is a centre of a sphere
    is_remaining = cp.ones((data_gpu_c0.shape[0],),dtype=bool)

    #define variable for hyperspheres
    n_hyperspheres_c0 = 0

    # Initialize a list to store the shapes of distance matrices
    distance_matrix_shapes = []
 
    #iterate over sample from largest to lowest radius
    for i in furthest_distance_index:

        #check if all samples are covered
        if not cp.any(is_remaining):
            #print(f'Stopped because all samples are removed')
            break

        # if sample not covered
        if is_remaining[i]:

            #count +1 hypersphere
            n_hyperspheres_c0 += 1
            
            #determine distances 
            distance_matrix_c0 = pairwise_distances(data_gpu_c0[i,:].reshape(1,-1),data_gpu_c0[is_remaining,:])

            #determine points within radius and remove
            mask               = distance_matrix_c0 > furthest_distances_c0[i]
            is_remaining[is_remaining] = cp.logical_and(mask.flatten(),is_remaining[is_remaining])
            
            #remove current sample
            is_remaining[i]    = False

            ############### stopping loop ###########

            # append the shape of the distance matrix
            distance_matrix_shapes.append(distance_matrix_c0.shape[1])

            # Keep only the last 100 shapes
            distance_matrix_shapes = distance_matrix_shapes[-100:]

            if np.mean(np.abs(np.diff(np.array(distance_matrix_shapes)))) < 1.5 and len(distance_matrix_shapes) == 100:

                #print('Stopped loop due to only isolated samples remaining')

                n_hyperspheres_c0 += cp.sum(is_remaining)

                break

    return n_hyperspheres_c0 / data_gpu_c0.shape[0]

def compute_T1_imbalanced_gpu(data,labels):

    # Transfer data and labels to GPU
    data_gpu = cp.array(data).astype(cp.float32)
    labels_gpu = cp.array(labels).astype(cp.int16)

    #separate data
    data_gpu_c0 = data_gpu[labels_gpu==0,:]
    data_gpu_c1 = data_gpu[labels_gpu==1,:]

    #compute radii variable

    #compute distance between c0 and c1
    knn_1 = NearestNeighbors(n_neighbors=2) # Find the single closest neighbor
    knn_1.fit(data_gpu_c1)

    furthers_distances_c0,_  = knn_1.kneighbors(data_gpu_c0,1)
    furthers_distances_c0    = furthers_distances_c0[:,0]
        
    #same but inverse class order
    knn_2 = NearestNeighbors(n_neighbors=2) 
    knn_2.fit(data_gpu_c0)  

    furthers_distances_c1,_  = knn_2.kneighbors(data_gpu_c1,2)
    furthers_distances_c1    = furthers_distances_c1[:,0]

    #sort C0 and c1 radius
    furthers_distances_index_c0 = cp.flip(cp.argsort(furthers_distances_c0))
    furthers_distances_index_c1 = cp.flip(cp.argsort(furthers_distances_c1))

    #compute metric
    T1_c0 = T1_class_loop(data_gpu_c0,furthers_distances_index_c0,furthers_distances_c0)
    T1_c1 = T1_class_loop(data_gpu_c1,furthers_distances_index_c1,furthers_distances_c1)
    T1    = (T1_c0+T1_c1)/2

    return float(T1_c0),float(T1_c1), float(T1)

def hypersphere_T1(data,labels):

    # Transfer data and labels to GPU
    data_gpu = cp.array(data).astype(cp.float32)
    labels_gpu = cp.array(labels).astype(cp.int16)

    #separate data
    data_gpu_c0 = data_gpu[labels_gpu==0,:]
    data_gpu_c1 = data_gpu[labels_gpu==1,:]

    #compute radii variable

    #compute distance between c0 and c1
    knn_1 = NearestNeighbors(n_neighbors=2) # Find the single closest neighbor
    knn_1.fit(data_gpu_c1)

    furthers_distances_c0,_  = knn_1.kneighbors(data_gpu_c0,1)
    furthers_distances_c0    = furthers_distances_c0[:,0]
        
    #same but inverse class order
    knn_2 = NearestNeighbors(n_neighbors=2) 
    knn_2.fit(data_gpu_c0)  

    furthers_distances_c1,_  = knn_2.kneighbors(data_gpu_c1,2)
    furthers_distances_c1    = furthers_distances_c1[:,0]

    #merge variables to create radii
    radii        = cp.concatenate([furthers_distances_c0,furthers_distances_c1])

    #determine remaining hyperspheres
    remaining_hypershperes = cp.ones((len(radii),),dtype=bool)
    chunk_size = 10**3


    for i in range(0,data_gpu_c0.shape[0],chunk_size):

        end = min(i + chunk_size, data_gpu_c0.shape[0])

        #print(f'Analyzed iteration between {i} and {end}')

        distance_matrix_c0    = pairwise_distances(data_gpu_c0[i:end],data_gpu_c0)

        mask = (distance_matrix_c0 < radii[None,:data_gpu_c0.shape[0]]) & (radii[None,:data_gpu_c0.shape[0]] > radii[i:end,None])
        remaining_hypershperes[i:end] = ~cp.any(mask,axis=1)
 

    for i in range(0,data_gpu_c1.shape[0],chunk_size):

        end = min(i + chunk_size, data_gpu_c1.shape[0])

        distance_matrix_c1    = pairwise_distances(data_gpu_c1[i:end],data_gpu_c1)

        mask = (distance_matrix_c1 < radii[None,data_gpu_c0.shape[0]:]) & (radii[None,data_gpu_c0.shape[0]:] > radii[i+data_gpu_c0.shape[0]:end+data_gpu_c0.shape[0],None])
        remaining_hypershperes[i+data_gpu_c0.shape[0]:end+data_gpu_c0.shape[0]] = ~cp.any(mask,axis=1)

        #print(f'Analyzed iteration between {i+data_gpu_c0.shape[0]} and {end+data_gpu_c0.shape[0]}')

    
    T1_c0 = cp.sum(remaining_hypershperes[:data_gpu_c0.shape[0]]) / data_gpu_c0.shape[0]
    T1_c1 = cp.sum(remaining_hypershperes[data_gpu_c0.shape[0]:]) / data_gpu_c1.shape[0]
    T1    = (T1_c0+T1_c1)/2

    return float(T1_c0), float(T1_c1), float(T1)

def Raug(data_gpu,labels_gpu,target_labels,k,delta):


    #compute nearest neighbours between target class and remaining data points
    knn_1 = NearestNeighbors(n_neighbors=k+1) 
    knn_1.fit(data_gpu)

    _, closest_indexes  = knn_1.kneighbors(data_gpu[target_labels,:],k+1)

    #remove self-comparisons
    closest_indexes     = closest_indexes[:,1:]

    #determine counts
    counts = cp.sum(labels_gpu[closest_indexes] != labels_gpu[target_labels][0],axis=1)

    return cp.sum(counts > delta)

def compute_Raug_imbalanced_gpu(data,labels,k=5,delta=2):

    # Transfer data and labels to GPU
    data_gpu = cp.array(data).astype(cp.float32)
    labels_gpu = cp.array(labels).astype(cp.int16)

    #get index of labels for both classes
    labels_c0 = labels_gpu == 0
    labels_c1 = labels_gpu == 1

    #compute R(minority) and R(majority)
    R_maj = Raug(data_gpu,labels_gpu,labels_c0,k,delta)
    R_min = Raug(data_gpu,labels_gpu,labels_c1,k,delta)

    #compute final metric
    IR = float((cp.sum(labels_c0)/cp.sum(labels_c1)))
    Raug_final = (R_maj + IR * R_min) / (IR + 1)
    
    return float(R_maj / cp.sum(labels_c0)), float(R_min / cp.sum(labels_c1)), float(Raug_final)

def compute_bayes_imbalance_ratio(data,labels,k=5,search_depth=100):

    # Transfer data and labels to GPU
    data_gpu = cp.array(data).astype(cp.float32)
    labels_gpu = cp.array(labels).astype(cp.int16)

    #avoid errors due to search depth being over the number of samples
    if data_gpu.shape[0] < search_depth:
        print('changed search depth')
        search_depth = data_gpu.shape[0] - 1 

    # select minority sample data
    data_gpu_c0 = data_gpu[labels_gpu==1,:]

    #compute nearest neighbours (100)
    knn_1 = NearestNeighbors(n_neighbors=search_depth+1) 
    knn_1.fit(data_gpu)

    _, closest_indexes  = knn_1.kneighbors(data_gpu_c0,search_depth+1)

    #remove self-comparisons
    closest_indexes     = closest_indexes[:,1:]

    # Compute M values for minority instances
    M_values_cumulative = cp.cumsum(labels_gpu[closest_indexes] == 1, axis=1)

    M_values = cp.zeros((M_values_cumulative.shape[0]))
    M_values = M_values_cumulative[:,k-1]

    #expansion needed
    expansion_needed = M_values == k
    M_values[expansion_needed] = cp.argmin(cp.diff(M_values_cumulative[expansion_needed,:],axis=1),axis=1)

    #if no majority samples is found
    M_values[(expansion_needed) & (M_values==0)] = search_depth

    # Initialize counts
    counts = cp.zeros(M_values.shape, dtype=int)
    counts[expansion_needed] = M_values[expansion_needed] + 1 - k

    # Compute fp', fp, fn values for all instances
    fp_values = ((k + counts) - M_values) / (k + counts)
    fp_balanced_values = (cp.sum(labels_gpu==0) / cp.sum(labels_gpu==1)) * fp_values

    fn_values = M_values / (k + counts)

    # Compute IBI^3 index for all instances
    IBI = (fp_balanced_values / (fp_balanced_values + fn_values)) - (fp_values / (fp_values + fn_values))
    
    # Compute BI^3
    BI = np.mean(IBI)

    return float(BI)


#-------------------------------
# regular

def compute_global_L1_gpu(data, labels):
    """
    Compute the global L1 metric for the dataset.
    """
    data_gpu = cp.array(data).astype(cp.float32)
    labels_gpu = cp.array(labels).astype(cp.int32)

    clf = LinearSVC()
    clf.fit(data_gpu, labels_gpu)

    predictions = clf.predict(data_gpu).astype(cp.int32)
    incorrect = predictions != labels_gpu

    # Absolute distances for all incorrect samples
    distances = cp.abs(clf.decision_function(data_gpu)[incorrect])
    L1 = cp.sum(distances) / data_gpu.shape[0]

    return float(1 / (1 + L1))

def compute_global_error_gpu(data, labels):
    """
    Compute the global error rate of a linear classifier on the dataset.
    """
    data_gpu = cp.array(data).astype(cp.float32)
    labels_gpu = cp.array(labels).astype(cp.int32)

    clf = LinearSVC()
    clf.fit(data_gpu, labels_gpu)

    predictions = clf.predict(data_gpu)
    error = cp.mean(predictions != labels_gpu)

    return float(error)

def compute_global_synthetic_error_gpu(data, labels):
    """
    Compute the global error on interpolated synthetic samples.
    """
    data_gpu = cp.array(data).astype(cp.float32)
    labels_gpu = cp.array(labels).astype(cp.int32)

    clf = LinearSVC()
    clf.fit(data_gpu, labels_gpu)

    # Generate synthetic data by interpolation within each class
    synthetic_samples = []
    synthetic_labels = []
    for label in cp.unique(labels_gpu):
        idx = cp.where(labels_gpu == label)[0]
        if len(idx) < 2:
            continue
        n = len(idx)
        a = cp.random.choice(idx, n, replace=True)
        b = cp.random.choice(idx, n, replace=True)
        alphas = cp.random.rand(n, 1)
        synth = alphas * data_gpu[a] + (1 - alphas) * data_gpu[b]
        synthetic_samples.append(synth)
        synthetic_labels.append(cp.full(n, label, dtype=cp.int32))

    if not synthetic_samples:
        return None

    test_data = cp.concatenate(synthetic_samples, axis=0)
    test_labels = cp.concatenate(synthetic_labels, axis=0)

    predictions = clf.predict(test_data)
    error = cp.mean(predictions != test_labels)

    return float(error)