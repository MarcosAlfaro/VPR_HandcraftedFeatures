# run train_exp1.py, train_exp2.py, train_exp3.py


#!/bin/bash

# python3 exp1_train.py

# # Verificar si el primer script se ejecutó correctamente
# if [ $? -ne 0 ]; then
#   echo "Error al ejecutar exp1_train.py"
#   exit 1
# fi

# python3 exp2_train.py

# # Verificar si el primer script se ejecutó correctamente
# if [ $? -ne 0 ]; then
#   echo "Error al ejecutar exp2_train.py"
#   exit 1
# fi


# python3 exp3_train.py

# # Verificar si el primer script se ejecutó correctamente
# if [ $? -ne 0 ]; then
#   echo "Error al ejecutar exp3_train.py"
#   exit 1
# fi

python3 exp3_lf_train_cold.py

# Verificar si el primer script se ejecutó correctamente
if [ $? -ne 0 ]; then
  echo "Error al ejecutar exp3_lf_train_cold.py"
  exit 1
fi

python3 exp3_lf_train_360loc.py

# Verificar si el primer script se ejecutó correctamente
if [ $? -ne 0 ]; then
  echo "Error al ejecutar exp3_lf_train_360loc.py"
  exit 1
fi



#python3 exp2_test_CL.py
#
## Verificar si el primer script se ejecutó correctamente
#if [ $? -ne 0 ]; then
#  echo "Error al ejecutar exp2_test_CL.py"
#  exit 1
#fi
