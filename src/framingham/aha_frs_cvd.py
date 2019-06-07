import numpy as np
import warnings
#white women
BETA_WOMEN_W = np.array([-29.799,        # natrual log age ln age
                         4.884,          # ln age squared
                         13.540,         # ln total Cholesterol (mg/dL)
                        -3.114,          # Ln Age×Ln Total Cholesterol
                        -13.578,         # Ln HDL–C
                         3.149,          # Ln Age×Ln
                         2.019,          # log treated systolic BP (mm Hg)
                         0,         # log Age×log treated systolic BP
                         1.957,          # log untreated systolic BP
                         0,         # log Age×log untreated systolic BP
                         7.574,          # smoking (1=yes,0=no)
                         -1.665,         # log age× smoking
                         0.661           # diabets
])

#african american women
BETA_WOMEN_B = np.array([17.114,        # natrual log age ln age
                         0,          # ln age squared
                         0.94,         # ln total Cholesterol (mg/dL)
                         0,          # Ln Age×Ln Total Cholesterol
                        -18.920,         # Ln HDL–C
                         4.475,          # Ln Age×Ln
                         29.291,          # log treated systolic BP (mm Hg)
                         -6.432,         # log Age×log treated systolic BP
                         27.82,          # log untreated systolic BP
                         -6.087,         # log Age×log untreated systolic BP
                         0.691,          # smoking (1=yes,0=no)
                         0,         # log age× smoking
                         0.874           # diabets
])

#white men
BETA_MEN_W = np.array([12.344,        # natrual log age ln age
                         11.853,         # ln total Cholesterol (mg/dL)
                         -2.664,          # Ln Age×Ln Total Cholesterol
                         -7.99,         # Ln HDL–C
                         1.769,          # Ln Age×Ln HDL-C
                         1.797,          # log treated systolic BP (mm Hg)
                         1.764,          # log untreated systolic BP
                         0.691,          # smoking (1=yes,0=no)
                         0,         # log age× smoking
                         0.658           # diabets
])

#white men
BETA_MEN_B = np.array([2.469,        # natrual log age ln age
                         0.302,         # ln total Cholesterol (mg/dL)
                         0,          # Ln Age×Ln Total Cholesterol
                         -0.307,         # Ln HDL–C
                         0,          # Ln Age×Ln HDL-C
                         1.916,          # log treated systolic BP (mm Hg)
                         1.809,          # log untreated systolic BP
                         0.549,          # smoking (1=yes,0=no)
                         0,         # log age× smoking
                         0.645           # diabets
])
# survival rate baseline
SURV_WOMEN_W=0.9665
SURV_WOMEN_B=0.9553
SURV_MEN_W=0.9144
SURV_MEN_B=0.8954



def _calc_frs(X,beta):
    sum = np.sum(X.dot(beta))
    return sum

def frs(gender='F', age=55, tchol=213, hdlc=50, sbp=120, smoking=0, diab=0, ht_treat=False, race='W',time=10):
    """
    :param gender: 'F' or 'M'
    :param age:
    :param tchol: total cholesterol
    :param hdlc:
    :param sbp: blood pressue
    :param smoking: 0 or 1
    :param diab: 0 or 1 (can be more than 1)
    :param ht_treat:0 or 1
    :param race:
    :param time: 10
    :return:
    """
    if time < 1 or time > 10:
        raise ValueError('Risk can only be calculated for 1 to 10 year time horizon')
    # try:

        #warnings.warn(Warning())
    # except RuntimeWarning:
    #     print('Raised')

    if gender.upper() == 'F':
        X_women = np.array([np.log(age), np.square(np.log(age)), np.log(tchol), np.log(age) * np.log(tchol),
                            np.log(hdlc), np.log(age) * np.log(hdlc), np.log(sbp) * bool(ht_treat),
                            np.log(age) * np.log(sbp) * bool(ht_treat), np.log(sbp) * (1 - bool(ht_treat)),
                            np.log(age) * np.log(sbp) * (1 - bool(ht_treat)), bool(smoking),
                            np.log(age) * bool(smoking), bool(diab)])
        if race.upper() == 'W':
            return _calc_frs(X_women, BETA_WOMEN_W)
        elif race.upper() =='B':
            return _calc_frs(X_women, BETA_WOMEN_B)
        else:
            raise ValueError('Race must be specified as W or B')
    elif gender.upper() == 'M':

        X_men = np.array([np.log(age), np.log(tchol), np.log(age) * np.log(tchol),
                          np.log(hdlc), np.log(age) * np.log(hdlc), np.log(sbp) * bool(ht_treat),
                          np.log(sbp) * (1 - bool(ht_treat)), bool(smoking),
                          np.log(age) * bool(smoking), bool(diab)])
        if race.upper() == 'W':
            return _calc_frs(X_men, BETA_MEN_W)
        elif race.upper() == 'B':
            return _calc_frs(X_men, BETA_MEN_B)
        else:
            raise ValueError('Race must be specified as W or B')
    else:
        raise ValueError('Gender must be specified as M or F')

def estimiate_risk(ind_frs, mean_frs, gender='F',race='W'):
    if gender.upper() == 'F':
        if race.upper() == 'W':
            return 1 - np.power(SURV_WOMEN_W, np.exp(ind_frs-mean_frs))
        elif race.upper() == 'B':
            return 1 - np.power(SURV_WOMEN_B, np.exp(ind_frs - mean_frs))
        else:
            raise ValueError('Race must be specified as W or B')
    elif gender.upper() == 'M':
        if race.upper() == 'W':
            return 1 - np.power(SURV_MEN_W, np.exp(ind_frs - mean_frs))
        elif race.upper() == 'B':
            return 1 - np.power(SURV_MEN_B, np.exp(ind_frs - mean_frs))
        else:
            raise ValueError('Race must be specified as W or B')
    else:
        raise ValueError('Gender must be specified as M or F')
    return 0

if __name__ == '__main__':
   #  # score = frs(gender='F', age=55, tchol=213, hdlc=50, sbp=120, smoking=0, diab=0, ht_treat=False, race='W',time=10)
   #  X = ['F', 55, 213, 50, 120, 0, 0, False, 'W', 10]
   #  score = frs(*X)
   #  print(score) #expected  -29.67
   #  risk = estimiate_risk(score, mean_frs=-29.18, gender='F',race='W') #expected 0.021
   #  print(risk)
   #
   #  X = ['M', 55, 213, 50, 120, 0, 0, False, 'W', 10]
   #  score = frs(*X)
   #  print(score)  # expected  60.69
   #  risk = estimiate_risk(score, mean_frs=61.18, gender='M', race='W')  # expected 0.0053
   #  print(risk)
   #
   # # score = frs(gender='F', age=55, tchol=213, hdlc=50, sbp=120, smoking=0, diab=0, ht_treat=False, race='W',time=10)
   #  X = ['F', 55, 213, 50, 120, 0, 0, False, 'B', 10]
   #  score = frs(*X)
   #  print(score) #expected  -29.67
   #  risk = estimiate_risk(score, mean_frs=86.61, gender='F',race='B') #expected 0.021
   #  print(risk)
   #
   #  X = ['M', 55, 213, 50, 120, 0, 0, False, 'B', 10]
   #  score = frs(*X)
   #  print(score)  # expected  60.69
   #  risk = estimiate_risk(score, mean_frs=19.54, gender='M', race='B')  # expected 0.0053
   #  print(risk)
   #
   #  X = ['F', 68, 180, 90, 151, 0, 1, 352.0,'B',10]
   #  score = frs(*X)
   #  print(score)
   #  risk = estimiate_risk(score, mean_frs=86.61, gender='F', race='B')  # expected 0.021
   #  print(risk)

    X = ['F', 60, 220, 59, 150, 0, 0, False, 'W', 10]
    score = frs(*X)
    print(score)
    risk = estimiate_risk(score, mean_frs=-29.18, gender='F', race='W')  # expected 0.021
    print(risk)

    X = ['M', 35, 200, 48, 106, 0, 0, False, 'W', 10]
    score = frs(*X)
    print(score)
    risk = estimiate_risk(score, mean_frs=61.68, gender='M', race='W')  # expected 0.021
    print(risk)



