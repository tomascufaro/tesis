from database import Database

db = Database("iemocap")

(
    dataset_x,
    dataset_y,
    _,
    _,
    _,
    _,
) = db.get_datasets()


def get_outliers(x):
    # Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(x.T)
    X = rng.multivariate_normal(mean=np.mean(x, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_  # robust covariance metric
    robust_mean = cov.location_  # robust mean
    inv_covmat = sp.linalg.inv(mcd)  # inverse covariance metric

    # Robust M-Distance
    x_minus_mu = x - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())

    # Flag as outlier
    outlier = []
    C = np.sqrt(
        stats.chi2.ppf((1 - 0.001), df=19)
    )  # degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md


positive_indexes = np.where(np.array(dataset_y) == 1)
negative_indexes = np.where(np.array(dataset_y) == 0)
negative_cases = dataset_x[negative_indexes]
positive_cases = dataset_x[positive_indexes]

positive_outliers = get_outliers(positive_cases)
print(len(positive_outliers))
negative_outliers = get_outliers(negative_cases)
print(len(negative_outliers))

outlier_indexes = np.concatenate((positive_outliers, negative_outliers), axis=0)

ids = db.dataset_no_aug.loc[outlier_indexes]["_id"]
print(ids)


"""
def robust_outliers(x):
    #Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(x.T)
    X = rng.multivariate_normal(mean=np.mean(x, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_ #robust covariance metric
    robust_mean = cov.location_  #robust mean
    inv_covmat = sp.linalg.inv(mcd) #inverse covariance metric
    
    #Robust M-Distance
    x_minus_mu = x - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())
    
    #Flag as outlier
    outlier = []
    C = np.sqrt(stats.chi2.ppf((1-0.001), df=19))#degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md

def outliers(x):
    mu = np.mean(x, axis=0)
    x_minus_mu = x - mu
    cov = np.cov(x.T)
    inv_cov = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_cov) 
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())
    #Flag as outlier
    outlier = []
    #Cut-off point
    C = np.sqrt(stats.chi2.ppf((1-0.001), df=19))    #degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier


positive_indexes = np.where(np.array(full_Y) == 1)
negative_indexes = np.where(np.array(full_Y) == 0)
negative_cases = full_x[negative_indexes]
positive_cases = full_x[positive_indexes]

positive_outliers = outliers(positive_cases)
print(len(positive_outliers))
negative_outliers = outliers(negative_cases)
print(len(negative_outliers))

robust_positive_outliers, _ = robust_outliers(positive_cases)
print(len(robust_positive_outliers))
robust_negative_outliers, _ = robust_outliers(negative_cases)
print(len(robust_negative_outliers))

positive_no_outliers = np.delete(positive_cases, robust_positive_outliers,0)
negative_no_outliers = np.delete(negative_cases, robust_negative_outliers,0)

full_no_outliers = np.concatenate((positive_no_outliers, negative_no_outliers),axis=0)
full_no_outliers.shape

y = list(np.ones(len(positive_no_outliers), dtype=int)) + list(np.zeros(len(negative_no_outliers), dtype=int))
len(y)

x, y = shuffle(full_no_outliers, y, random_state=0)

general_outliers, _ = robust_outliers(x)
x_no_outliers = np.delete(x, general_outliers,0)
y_no_outliers = np.delete(y, general_outliers,0)
x_no_outliers.shape

positives = np.count_nonzero(np.array(y_no_outliers) == 1)
negatives = np.count_nonzero(np.array(y_no_outliers) == 0)
print(positives, negatives)

x_downsampled, y_downsampled = downsample(x_no_outliers, y_no_outliers, 0, 0.62)
enterface_positives = np.count_nonzero(np.array(y_downsampled) == 1)
enterface_negatives = np.count_nonzero(np.array(y_downsampled) == 0)
print(enterface_negatives, enterface_positives)
"""
