import numpy as np
array=np.array

seeds=[66,3391858867,3615040951,1284975474,3771498045]

data=[
    (2, 0.9954745025337225, array([ 4.06714136e+00,  3.14158197e+00,  3.14158665e+00,  3.14159257e+00,
            1.56978577e+00,  3.14158885e+00,  3.76566166e+00,  4.71236149e+00,
            6.23135139e+00,  4.71239607e+00,  1.61113698e+00,  4.80685321e+00,
            1.72265278e+00,  1.56696387e+00,  4.34247748e+00,  2.63933131e+00,
            6.50348554e+00,  4.89978978e+00,  3.19003282e+00,  2.13439393e+00,
            3.14148583e+00,  5.20247533e+00,  3.91909680e+00,  4.07964177e+00,
            3.23382573e+00,  1.37521314e+00,  6.20064462e+00,  4.00819925e+00,
            4.91864887e+00,  8.34247056e-01,  1.55875537e+00,  1.45630535e+00,
            4.03958528e+00,  5.71111162e-01,  1.23245489e+00,  3.28926884e+00,
            6.28321244e+00,  3.94247568e+00,  1.02143196e+00,  1.27524466e+00,
            2.50834230e+00,  4.37108530e+00,  3.20210649e+00,  2.83126993e+00,
            1.03316353e+00,  3.93794250e+00,  3.94337572e+00,  1.87681866e+00,
            9.58636077e-05,  1.85960417e+00,  6.27598247e+00,  4.23083348e+00,
            5.67715965e+00, -8.88481815e-02,  6.33422900e+00,  1.76569679e+00,
            2.19797674e+00,  7.62706263e+00,  6.12261457e-01,  7.26762820e+00,
            6.24172175e-01,  1.40005500e+00,  4.71238301e+00,  3.14159203e+00,
            4.71238419e+00,  4.71239979e+00,  4.71170956e+00,  3.14159249e+00,
            3.14159269e+00,  1.03724004e+00, -1.71825538e-01,  4.19460426e+00,
            4.50444918e+00,  4.24352465e+00,  3.11641748e+00,  3.14127634e+00,
            1.57078346e+00,  6.28318500e+00,  2.64574257e-06,  3.14157389e+00,
            4.16483085e+00,  1.58478551e+00,  1.57080368e+00,  1.57082774e+00,
            6.28336232e+00,  5.92123183e+00,  6.64255231e+00,  3.14159824e+00,
           -1.57079589e+00,  8.74409622e-01,  4.89097727e+00,  6.28318758e+00,
            2.12194574e-01,  2.72140710e+00, -1.63610325e+00, -6.63245663e-01,
            1.12525214e+00,  1.57085306e+00,  1.57079815e+00,  8.64241482e-01,
            1.57079249e+00,  3.14160048e+00,  3.14159240e+00,  3.14159232e+00,
            7.82792458e-01,  2.52447857e+00,  6.81162583e-01,  1.15695480e-01,
            4.61673811e+00, -3.27483281e-01,  3.36037843e+00,  7.75557964e-01,
            6.28318699e+00,  1.57078551e+00, -8.64223791e-01,  5.39303194e+00,
            1.67820580e+00,  4.71239330e+00, -7.85383009e-01, -1.57079949e+00,
            4.51601745e+00,  1.59904182e+00,  6.28317958e+00,  1.57079224e+00,
            3.14906198e+00,  2.45295616e+00,  3.14700800e+00,  3.15136565e+00,
            1.70219716e+00,  4.81110407e+00,  1.23260503e+00, -9.11250519e-02,
            4.71240439e+00,  1.57079763e+00,  3.91716047e+00,  4.71238597e+00,
           -1.39672707e+00,  3.14159241e+00,  6.28318518e+00, -1.07827628e+00,
            3.33752489e+00,  2.38873191e+00,  3.85600853e+00,  2.41184417e+00,
            1.51076937e+00,  5.92302851e+00,  3.14159247e+00,  1.57079598e+00,
            4.71238617e+00,  6.28316994e+00,  2.93987025e+00,  2.43260251e+00,
            4.71238158e+00,  4.71239169e+00,  3.14159370e+00,  3.18605102e+00,
            6.13170823e+00,  4.71238833e+00,  6.28319340e+00,  3.97195833e+00,
            1.24304487e-02,  6.28317904e+00,  4.25799486e+00,  5.11437352e+00,
            4.78148358e+00,  3.69247715e+00,  2.05486527e+00,  3.76013693e-06,
            4.71239325e+00,  1.57079619e+00,  4.71238908e+00,  1.00466144e+01,
            4.58536893e+00,  5.60371095e+00,  9.43952686e+00,  3.14159158e+00,
            4.14908043e-06,  6.28319090e+00,  5.98129989e+00,  2.42714967e-01,
            1.11394513e+00])),

    (4, 0.9888160324383086, array([ 4.06714118e+00,  3.14159441e+00,  3.14159709e+00,  3.14158917e+00,
            1.57050609e+00,  3.14158688e+00,  3.76566237e+00,  4.71238673e+00,
            6.23134635e+00,  4.71239277e+00,  1.61114507e+00,  4.80686716e+00,
            1.72266073e+00,  1.56696469e+00,  4.34249571e+00,  2.63922453e+00,
            6.50348558e+00,  4.89979774e+00,  3.19003237e+00,  2.13436095e+00,
            3.14139904e+00,  5.20249124e+00,  3.91911390e+00,  4.07964385e+00,
            3.23384636e+00,  1.37521870e+00,  6.20064348e+00,  4.00817992e+00,
            4.91865808e+00,  8.34245061e-01,  1.55876510e+00,  1.45631317e+00,
            4.03959885e+00,  5.71113728e-01,  1.23246828e+00,  3.28927110e+00,
            6.28319084e+00,  3.94248970e+00,  1.02144302e+00,  1.27524836e+00,
            2.50835189e+00,  4.37108691e+00,  3.20211893e+00,  2.83128736e+00,
            1.03317003e+00,  3.93794481e+00,  3.94338246e+00,  1.87679390e+00,
            1.50826652e-04,  1.85943083e+00,  6.27598419e+00,  4.23080954e+00,
            5.67715494e+00, -8.88762715e-02,  6.33423710e+00,  1.76569643e+00,
            2.19795817e+00,  7.62706684e+00,  6.12236002e-01,  7.26764395e+00,
            6.24177132e-01,  1.40005960e+00,  4.71238648e+00,  3.14159567e+00,
            4.71239119e+00,  4.71241097e+00,  4.71214128e+00,  3.14159267e+00,
            3.14159262e+00,  1.03724265e+00, -1.71791336e-01,  4.19460253e+00,
            4.50445220e+00,  4.24282534e+00,  3.11642290e+00,  3.14128478e+00,
            1.57077401e+00,  6.28318568e+00, -7.28608094e-07,  3.14155573e+00,
            4.16483954e+00,  1.58479037e+00,  1.57079934e+00,  1.57081024e+00,
            6.28321327e+00,  5.92120560e+00,  6.64253101e+00,  3.14157285e+00,
           -1.57079224e+00,  8.74413966e-01,  4.89098579e+00,  6.28318188e+00,
            2.12207941e-01,  2.72141582e+00, -1.63540507e+00, -6.62544244e-01,
            1.12526031e+00,  1.57089264e+00,  1.57079663e+00,  8.64285291e-01,
            1.57081162e+00,  3.14159153e+00,  3.14159273e+00,  3.14159262e+00,
            7.82786336e-01,  2.52447643e+00,  6.81163535e-01,  1.15703492e-01,
            4.61743973e+00, -3.27471100e-01,  3.36038051e+00,  7.67008997e-01,
            6.28318504e+00,  1.57079217e+00, -8.64239216e-01,  5.39301875e+00,
            1.67820183e+00,  4.71239441e+00, -7.85325740e-01, -1.57079878e+00,
            4.51601014e+00,  1.59904029e+00,  6.28318713e+00,  1.57079755e+00,
            3.14906676e+00,  2.45295221e+00,  3.15190431e+00,  3.15137346e+00,
            1.70220342e+00,  4.81180402e+00,  1.23330763e+00, -9.11221055e-02,
            4.71242151e+00,  1.57079738e+00,  3.90860058e+00,  4.71238031e+00,
           -1.39661599e+00,  3.14159277e+00,  6.28318558e+00, -1.07827059e+00,
            3.33752324e+00,  2.38873069e+00,  3.85600372e+00,  2.41190295e+00,
            1.51077474e+00,  5.92303442e+00,  3.14159302e+00,  1.57079657e+00,
            4.71238919e+00,  6.28315158e+00,  2.93987187e+00,  2.43259955e+00,
            4.71233571e+00,  4.71238787e+00,  3.14159261e+00,  3.18605083e+00,
            6.13171172e+00,  4.71237195e+00,  6.28319184e+00,  3.97196749e+00,
            1.24360001e-02,  6.28317089e+00,  4.25800101e+00,  5.11437842e+00,
            4.78143158e+00,  3.69253468e+00,  2.05487258e+00, -6.89779436e-06,
            4.71238491e+00,  1.57079648e+00,  4.71238950e+00,  1.00472706e+01,
            4.58536893e+00,  5.60371728e+00,  9.43952802e+00,  3.14160215e+00,
           -8.35156970e-07,  6.28318300e+00,  5.98141396e+00,  2.42714410e-01,
            1.11394780e+00])),



    (6, 0.9803213112868919, array([ 4.06714501e+00,  3.14159215e+00,  3.14159254e+00,  3.14159381e+00,
            1.57055400e+00,  3.14158916e+00,  3.76564232e+00,  4.71237600e+00,
            6.23134412e+00,  4.71239686e+00,  1.61114294e+00,  4.80687037e+00,
            1.72265670e+00,  1.56697771e+00,  4.34250248e+00,  2.63918096e+00,
            6.50349043e+00,  4.89979740e+00,  3.19003126e+00,  2.13434726e+00,
            3.14136836e+00,  5.20248879e+00,  3.91917008e+00,  4.07964362e+00,
            3.23384147e+00,  1.37521612e+00,  6.20064062e+00,  4.00815640e+00,
            4.91865947e+00,  8.34247565e-01,  1.55877404e+00,  1.45630647e+00,
            4.03960242e+00,  5.71111214e-01,  1.23247193e+00,  3.28926759e+00,
            6.28319805e+00,  3.94248565e+00,  1.02144844e+00,  1.27525237e+00,
            2.50836062e+00,  4.37108028e+00,  3.20211837e+00,  2.83129980e+00,
            1.03316789e+00,  3.93794272e+00,  3.94337608e+00,  1.87686595e+00,
            1.60003392e-04,  1.85925297e+00,  6.27598282e+00,  4.23079582e+00,
            5.67715556e+00, -8.88894926e-02,  6.33423776e+00,  1.76570123e+00,
            2.19794195e+00,  7.62706445e+00,  6.12220709e-01,  7.26764742e+00,
            6.24175191e-01,  1.40005940e+00,  4.71238248e+00,  3.14159345e+00,
            4.71238969e+00,  4.71241248e+00,  4.71225808e+00,  3.14159245e+00,
            3.14159210e+00,  1.03723820e+00, -1.71780562e-01,  4.19459108e+00,
            4.50444267e+00,  4.24208807e+00,  3.11642185e+00,  3.14128295e+00,
            1.57077014e+00,  6.28318504e+00, -2.03026116e-07,  3.14154869e+00,
            4.16484086e+00,  1.58479584e+00,  1.57079950e+00,  1.57079239e+00,
            6.28311221e+00,  5.92119079e+00,  6.64251176e+00,  3.14159846e+00,
           -1.57080292e+00,  8.74426180e-01,  4.89099405e+00,  6.28318450e+00,
            2.12212530e-01,  2.72142278e+00, -1.63466507e+00, -6.61806121e-01,
            1.12525798e+00,  1.57090582e+00,  1.57079629e+00,  8.64131722e-01,
            1.57080330e+00,  3.14156078e+00,  3.14159232e+00,  3.14159229e+00,
            7.82774769e-01,  2.52448702e+00,  6.81164413e-01,  1.15700259e-01,
            4.61817575e+00, -3.27474043e-01,  3.36038000e+00,  7.59605673e-01,
            6.28317851e+00,  1.57079728e+00, -8.64030501e-01,  5.39301231e+00,
            1.67819262e+00,  4.71238474e+00, -7.85208585e-01, -1.57079638e+00,
            4.51602519e+00,  1.59905084e+00,  6.28318306e+00,  1.57078901e+00,
            3.14906878e+00,  2.45295296e+00,  3.15689468e+00,  3.15137237e+00,
            1.70220202e+00,  4.81254386e+00,  1.23404586e+00, -9.11234286e-02,
            4.71243198e+00,  1.57079526e+00,  3.90119115e+00,  4.71238685e+00,
           -1.39650674e+00,  3.14159242e+00,  6.28318515e+00, -1.07827003e+00,
            3.33752171e+00,  2.38873338e+00,  3.85600525e+00,  2.41200042e+00,
            1.51077499e+00,  5.92303169e+00,  3.14159240e+00,  1.57079592e+00,
            4.71238867e+00,  6.28314349e+00,  2.93987305e+00,  2.43259723e+00,
            4.71225102e+00,  4.71239239e+00,  3.14158847e+00,  3.18604924e+00,
            6.13171158e+00,  4.71238776e+00,  6.28317828e+00,  3.97196665e+00,
            1.24383346e-02,  6.28317783e+00,  4.25800029e+00,  5.11437716e+00,
            4.78133173e+00,  3.69263026e+00,  2.05486666e+00,  4.91616117e-06,
            4.71238951e+00,  1.57079592e+00,  4.71238910e+00,  1.00479543e+01,
            4.58536631e+00,  5.60371522e+00,  9.43952947e+00,  3.14159707e+00,
           -1.49132359e-06,  6.28318801e+00,  5.98150226e+00,  2.42714970e-01,
            1.11394675e+00])),

    (8, 0.9702615967631429, array([ 4.06714303e+00,  3.14159190e+00,  3.14159152e+00,  3.14159445e+00,
            1.57081401e+00,  3.14159598e+00,  3.76563916e+00,  4.71235949e+00,
            6.23134496e+00,  4.71239735e+00,  1.61115638e+00,  4.80687861e+00,
            1.72265117e+00,  1.56702424e+00,  4.34249719e+00,  2.63912476e+00,
            6.50349060e+00,  4.89979909e+00,  3.19008516e+00,  2.13432155e+00,
            3.14139847e+00,  5.20247977e+00,  3.91918563e+00,  4.07964377e+00,
            3.23383473e+00,  1.37521594e+00,  6.20062715e+00,  4.00814650e+00,
            4.91866410e+00,  8.34255932e-01,  1.55877194e+00,  1.45629680e+00,
            4.03959440e+00,  5.71112274e-01,  1.23247377e+00,  3.28925742e+00,
            6.28318235e+00,  3.94248488e+00,  1.02145045e+00,  1.27526014e+00,
            2.50836510e+00,  4.37107801e+00,  3.20212397e+00,  2.83131782e+00,
            1.03316518e+00,  3.93793758e+00,  3.94338163e+00,  1.87684255e+00,
            1.53069942e-04,  1.85916266e+00,  6.27598249e+00,  4.23079159e+00,
            5.67715596e+00, -8.88946142e-02,  6.33423684e+00,  1.76569967e+00,
            2.19793970e+00,  7.62706366e+00,  6.12216878e-01,  7.26764776e+00,
            6.24174869e-01,  1.40005974e+00,  4.71239697e+00,  3.14158893e+00,
            4.71238647e+00,  4.71242030e+00,  4.71238882e+00,  3.14159285e+00,
            3.14159278e+00,  1.03724346e+00, -1.71777226e-01,  4.19457445e+00,
            4.50445196e+00,  4.24137439e+00,  3.11642035e+00,  3.14128314e+00,
            1.57077441e+00,  6.28318533e+00,  2.21079122e-06,  3.14158586e+00,
            4.16483484e+00,  1.58478851e+00,  1.57082012e+00,  1.57083639e+00,
            6.28307407e+00,  5.92118566e+00,  6.64250896e+00,  3.14159087e+00,
           -1.57085655e+00,  8.74442046e-01,  4.89101068e+00,  6.28318696e+00,
            2.12199641e-01,  2.72141165e+00, -1.63395604e+00, -6.61095835e-01,
            1.12525545e+00,  1.57091352e+00,  1.57079904e+00,  8.64044456e-01,
            1.57082117e+00,  3.14149393e+00,  3.14159260e+00,  3.14159282e+00,
            7.82775780e-01,  2.52453626e+00,  6.81166449e-01,  1.15698931e-01,
            4.61888582e+00, -3.27474350e-01,  3.36037900e+00,  7.53204705e-01,
            6.28316656e+00,  1.57080269e+00, -8.63874498e-01,  5.39301147e+00,
            1.67819271e+00,  4.71235356e+00, -7.85077998e-01, -1.57079903e+00,
            4.51607154e+00,  1.59909892e+00,  6.28317198e+00,  1.57072954e+00,
            3.14906961e+00,  2.45295484e+00,  3.16181870e+00,  3.15137129e+00,
            1.70220377e+00,  4.81325213e+00,  1.23475623e+00, -9.11237135e-02,
            4.71239524e+00,  1.57078641e+00,  3.89483442e+00,  4.71239380e+00,
           -1.39642078e+00,  3.14159283e+00,  6.28318556e+00, -1.07827102e+00,
            3.33752285e+00,  2.38873968e+00,  3.85600575e+00,  2.41205762e+00,
            1.51077445e+00,  5.92303118e+00,  3.14159280e+00,  1.57079616e+00,
            4.71238393e+00,  6.28317986e+00,  2.93987313e+00,  2.43259698e+00,
            4.71212479e+00,  4.71239477e+00,  3.14157847e+00,  3.18604804e+00,
            6.13171189e+00,  4.71238708e+00,  6.28312486e+00,  3.97195796e+00,
            1.24452148e-02,  6.28318331e+00,  4.25799874e+00,  5.11437881e+00,
            4.78127442e+00,  3.69268688e+00,  2.05486740e+00, -1.42801937e-06,
            4.71239082e+00,  1.57079650e+00,  4.71238824e+00,  1.00486216e+01,
            4.58536670e+00,  5.60371480e+00,  9.43952813e+00,  3.14160568e+00,
            3.45960185e-06,  6.28318206e+00,  5.98160500e+00,  2.42714252e-01,
            1.11394504e+00])),

    (10, 0.9588828638038678, array([ 4.06714293e+00,  3.14159285e+00,  3.14159260e+00,  3.14159382e+00,
            1.57101242e+00,  3.14159351e+00,  3.76562755e+00,  4.71238955e+00,
            6.23133651e+00,  4.71239227e+00,  1.61114805e+00,  4.80687427e+00,
            1.72265686e+00,  1.56699722e+00,  4.34249935e+00,  2.63905437e+00,
            6.50348808e+00,  4.89979762e+00,  3.19004892e+00,  2.13429608e+00,
            3.14139971e+00,  5.20248835e+00,  3.91922856e+00,  4.07964159e+00,
            3.23384385e+00,  1.37521718e+00,  6.20063913e+00,  4.00812310e+00,
            4.91865877e+00,  8.34251604e-01,  1.55877265e+00,  1.45630497e+00,
            4.03960304e+00,  5.71115197e-01,  1.23248436e+00,  3.28926199e+00,
            6.28319113e+00,  3.94248197e+00,  1.02146231e+00,  1.27525325e+00,
            2.50838206e+00,  4.37107333e+00,  3.20211978e+00,  2.83130425e+00,
            1.03316829e+00,  3.93793848e+00,  3.94337761e+00,  1.87684668e+00,
            1.47209386e-04,  1.85900871e+00,  6.27598320e+00,  4.23078874e+00,
            5.67715770e+00, -8.88988443e-02,  6.33423746e+00,  1.76570225e+00,
            2.19793571e+00,  7.62706510e+00,  6.12213439e-01,  7.26764818e+00,
            6.24175273e-01,  1.40005977e+00,  4.71238705e+00,  3.14159225e+00,
            4.71239301e+00,  4.71240607e+00,  4.71246218e+00,  3.14159291e+00,
            3.14159274e+00,  1.03724156e+00, -1.71772231e-01,  4.19458695e+00,
            4.50443957e+00,  4.24066413e+00,  3.11642056e+00,  3.14128321e+00,
            1.57077528e+00,  6.28318522e+00, -1.14004426e-06,  3.14155235e+00,
            4.16483647e+00,  1.58479110e+00,  1.57079983e+00,  1.57079915e+00,
            6.28302854e+00,  5.92118191e+00,  6.64250613e+00,  3.14159531e+00,
           -1.57080326e+00,  8.74428747e-01,  4.89099765e+00,  6.28318425e+00,
            2.12213166e-01,  2.72142583e+00, -1.63324421e+00, -6.60385190e-01,
            1.12525428e+00,  1.57088565e+00,  1.57079701e+00,  8.63857345e-01,
            1.57080116e+00,  3.14155345e+00,  3.14159277e+00,  3.14159279e+00,
            7.82777596e-01,  2.52458427e+00,  6.81165171e-01,  1.15698672e-01,
            4.61959665e+00, -3.27474641e-01,  3.36037904e+00,  7.47752377e-01,
            6.28317798e+00,  1.57079703e+00, -8.63607908e-01,  5.39301229e+00,
            1.67819375e+00,  4.71238462e+00, -7.84938955e-01, -1.57079585e+00,
            4.51611992e+00,  1.59914763e+00,  6.28318294e+00,  1.57078953e+00,
            3.14906749e+00,  2.45295313e+00,  3.16667254e+00,  3.15137265e+00,
            1.70220464e+00,  4.81396246e+00,  1.23546689e+00, -9.11234026e-02,
            4.71242670e+00,  1.57079476e+00,  3.88936241e+00,  4.71238855e+00,
           -1.39630801e+00,  3.14159293e+00,  6.28318544e+00, -1.07826924e+00,
            3.33752310e+00,  2.38873223e+00,  3.85600562e+00,  2.41212941e+00,
            1.51077429e+00,  5.92303239e+00,  3.14159258e+00,  1.57079652e+00,
            4.71238895e+00,  6.28315094e+00,  2.93987248e+00,  2.43259659e+00,
            4.71195864e+00,  4.71239317e+00,  3.14158859e+00,  3.18604913e+00,
            6.13171073e+00,  4.71239115e+00,  6.28317778e+00,  3.97196465e+00,
            1.24376616e-02,  6.28318018e+00,  4.25799954e+00,  5.11437935e+00,
            4.78120438e+00,  3.69275840e+00,  2.05486656e+00,  5.73050920e-06,
            4.71238959e+00,  1.57079647e+00,  4.71238985e+00,  1.00492821e+01,
            4.58536625e+00,  5.60371606e+00,  9.43952703e+00,  3.14159629e+00,
           -5.64094478e-07,  6.28318783e+00,  5.98171119e+00,  2.42713660e-01,
            1.11394544e+00]))
    ]