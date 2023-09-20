################################################################################
#                              EGADE Business School                           #
#                              Maestría en Finanzas                            #
#                                                                              #
#                       Tarea 2: Depuración de un Dataset                      #
################################################################################
# Objetivo: Preparar un dataset balanceado, eliminando datos faltantes,        #
# selección de features análizando sus características                         #                                    
#                                                                              #
################################################################################ 

# Carga de los paquetes

library(moments)
library(dplyr)
library(ggplot2)
library(visdat)
library(naniar)
library(tidyr)
library(caTools)
library(caret)
library(broom)
library(class)
library(gmodels)
library(corrplot)

# Carga de los datos
LC <- read.csv(file.choose())
# Para guarda el dataset en formato *.Rda
save(LC, file="LC.Data.Rda")
# Iniciamos con el análisis de dataset
# Para conocer la estructura, variables y valores del data frame
glimpse(LC)
str(LC)

################################################################################
##### Missing Values #####
# Realizamos el análisis de valores faltantes en el dataset.
#Missing Values Count
LC %>% 
  n_miss()

#Proporci?n
LC %>% 
  prop_miss()

#An?lisis de la base
miss.var <- LC %>% 
  miss_var_summary()
miss.var

#Visualizaci?n de valores faltantes
#LC %>% 
#  vis_miss()

################################################################################

# Reducimos el número de features a 92 considerando aquellos que tienen el mayor 
# número de datos, con un criterio de menos del 10% de valores faltantes.

LCR <- LC %>%
  select(mths_since_recent_inq,
         num_tl_120dpd_2m,
         mo_sin_old_il_acct,
         bc_util,
         percent_bc_gt_75,
         bc_open_to_buy,
         mths_since_recent_bc,
         pct_tl_nvr_dlq,
         avg_cur_bal,
         mo_sin_old_rev_tl_op,
         mo_sin_rcnt_rev_tl_op,
         num_rev_accts,
         tot_coll_amt,
         tot_cur_bal,
         total_rev_hi_lim,
         mo_sin_rcnt_tl,
         num_accts_ever_120_pd,
         num_actv_bc_tl,
         num_actv_rev_tl,
         num_bc_tl,
         num_il_tl,
         num_op_rev_tl,
         num_rev_tl_bal_gt_0,
         num_tl_30dpd,
         num_tl_90g_dpd_24m,
         num_tl_op_past_12m,
         tot_hi_cred_lim,
         total_il_high_credit_limit,
         num_bc_sats,
         num_sats,
         acc_open_past_24mths,
         mort_acc,
         total_bal_ex_mort,
         total_bc_limit,
         revol_util,
         dti,
         pub_rec_bankruptcies,
         collections_12_mths_ex_med,
         chargeoff_within_12_mths,
         tax_liens,
         inq_last_6mths,
         delinq_2yrs,
         open_acc,
         pub_rec,
         total_acc,
         acc_now_delinq,
         delinq_amnt,
         annual_inc,
         loan_amnt,
         funded_amnt,
         funded_amnt_inv,
         int_rate,
         installment,
         fico_range_low,
         fico_range_high,
         revol_bal,
         out_prncp,
         out_prncp_inv,
         total_pymnt,
         total_pymnt_inv,
         total_rec_prncp,
         total_rec_int,
         total_rec_late_fee,
         recoveries,
         collection_recovery_fee,
         last_pymnt_amnt,
         last_fico_range_high,
         last_fico_range_low,
         policy_code,
         id,
         term,
         grade,
         sub_grade,
         emp_title,
         emp_length,
         home_ownership,
         verification_status,
         issue_d,
         loan_status,
         pymnt_plan,
         purpose,
         title,
         zip_code,
         addr_state,
         earliest_cr_line,
         initial_list_status,
         last_pymnt_d,
         last_credit_pull_d,
         application_type,
         hardship_flag,
         disbursement_method,
         debt_settlement_flag)

# Nuevamente realizamos el análisis de valores faltantes, para determinar la
# efectividad de la reducción de features del dataset.
##### Missing Values #####
#Missing Values Count
# LCR = Lending Club Reduce
LCR %>% 
  n_miss()

#Proporci?n
LCR %>% 
  prop_miss()

#An?lisis de la base
LCR.miss.var <- LCR %>% 
  miss_var_summary()
LCR.miss.var

#Visualizaci?n de valores faltantes
#LCR.miss.var %>% 
#  vis_miss()

# Convertimos los valores vacio en NA's para después eliminarlos con la funcion 
# na_drop()
# LCR <- LCR[apply(LCR == "", NA, all),]
################################################################################

##### Tratamiento de valores faltantes #####

# Alternativa 1: Quitarlos
# Beneficios: La informaci?n es la real, no se est? alterando
#             Los resultados son v?lidos para la muestra
# Contras: Se puede perder una gran cantidad de informaci?n
#          Los resultados dejan de ser v?lidos para la poblaci?n (en algunos casos)
# Operaci?n: Uso del comando drop_na()

#En este caso optaremos por eleminar los valores faltantes del dataset
# LCS = Lending Club Selection
LCS <- LCR %>%
  drop_na()

#Verificamos que no existan valores faltantes.
#Visualizaci?n de valores faltantes
LCS %>% 
  n_miss()

#Proporci?n
LCS %>% 
  prop_miss()

glimpse(LCS)

# Observamos los valores de la variable categórica "loan_status"
LCS %>%
  select(loan_status) %>%
  table()

# como pudimos observar existen diferentes valores para esta variable, 
# sin embargo los valores "Charged Off" y "Fully Paid" son los que resultan 
# de interés para nuestro análisis. Filtramos estos valores y los almacenamos 
# en un nuevo dataset.
#LCST <- lending Club Selection Target
LCST <- LCS %>%
  filter(loan_status == "Charged Off" | loan_status == "Fully Paid")
# Observamos los valores resultantes de la variable categórica "loan_status" 
LCST %>%
  select(loan_status) %>%
  table()
glimpse(LCST)
save(LCST, file = "LCST.Rda")
################################################################################
#                           Selección de Features
################################################################################
# Realizamos la selección de 10 features para realizar el ajuste del modelo mas 
# el feature loan_status que es la variable de interés.
glimpse(LCST)

LCST.A <- LCST %>%
  select(loan_status,
         loan_amnt,
         int_rate,
         dti,
         fico_range_low,
         fico_range_high,
         term,
         tot_hi_cred_lim,
         total_acc,
         total_bal_ex_mort,
         total_bc_limit,
         )

glimpse(LCST.A)
str(LCST.A)
################################################################################
#                        Exploracion inicial del dataset                       #
################################################################################
# Feature 1
LCST.A %>% 
  ggplot(aes(x=loan_amnt))+
  geom_histogram(bins=100,
                 fill="blue")+
  labs(title = "Histograma Loan_Amnt",
       subtitle = "Lending Club",
       y="Conteo",
       x="Loan Amount")+
  theme_bw()

LCST.A %>% 
  ggplot(aes(x=loan_amnt))+
  geom_density(fill="blue")

LCST.A %>% 
  ggplot(aes(x=loan_amnt))+
  geom_boxplot(fill="blue")

# Feature 2
LCST.A %>% 
  ggplot(aes(x=int_rate))+
  geom_histogram(bins=100,
                 fill="blue")+
  labs(title = "Histograma int_rate",
       subtitle = "Lending Club",
       y="Conteo",
       x="Interest Rate")+
  theme_bw()

LCST.A %>% 
  ggplot(aes(x=int_rate))+
  geom_density(fill="blue")

LCST.A %>% 
  ggplot(aes(x=int_rate))+
  geom_boxplot(fill="blue")

# Feature 3

LCST.A %>% 
  ggplot(aes(x=dti))+
  geom_histogram(bins=100,
                 fill="blue")+
  labs(title = "Histograma dti",
       subtitle = "Lending Club",
       y="Conteo",
       x="DTI")+
  theme_bw()

LCST.A %>% 
  ggplot(aes(x=dti))+
  geom_density(fill="blue")

LCST.A %>% 
  ggplot(aes(x=dti))+
  geom_boxplot(fill="blue")

# Feature 4

LCST.A %>% 
  ggplot(aes(x=fico_range_low))+
  geom_histogram(bins=100,
                 fill="blue")+
  labs(title = "Histograma fico_range_low",
       subtitle = "Lending Club",
       y="Conteo",
       x="fico_range_low")+
  theme_bw()

LCST.A %>% 
  ggplot(aes(x=fico_range_low))+
  geom_density(fill="blue")

LCST.A %>% 
  ggplot(aes(x=fico_range_low))+
  geom_boxplot(fill="blue")

# Feature 5

LCST.A %>% 
  ggplot(aes(x=fico_range_high))+
  geom_histogram(bins=100,
                 fill="blue")+
  labs(title = "Histograma fico_range_high",
       subtitle = "Lending Club",
       y="Conteo",
       x="fico_range_high")+
  theme_bw()

LCST.A %>% 
  ggplot(aes(x=fico_range_high))+
  geom_density(fill="blue")

LCST.A %>% 
  ggplot(aes(x=fico_range_high))+
  geom_boxplot(fill="blue")

# Feature 6

LCST.A %>% 
  ggplot(aes(x=tot_hi_cred_lim))+
  geom_histogram(bins=100,
                 fill="blue")+
  labs(title = "Histograma tot_hi_cred_lim",
       subtitle = "Lending Club",
       y="Conteo",
       x="tot_hi_cred_lim")+
  theme_bw()

LCST.A %>% 
  ggplot(aes(x=tot_hi_cred_lim))+
  geom_density(fill="blue")

LCST.A %>% 
  ggplot(aes(x=tot_hi_cred_lim))+
  geom_boxplot(fill="blue")

# Feature 7

LCST.A %>% 
  ggplot(aes(x=total_acc))+
  geom_histogram(bins=100,
                 fill="blue")+
  labs(title = "Histograma total_acc",
       subtitle = "Lending Club",
       y="Conteo",
       x="total_acc")+
  theme_bw()

LCST.A %>% 
  ggplot(aes(x=total_acc))+
  geom_density(fill="blue")

LCST.A %>% 
  ggplot(aes(x=total_acc))+
  geom_boxplot(fill="blue")

# Feature 8

LCST.A %>% 
  ggplot(aes(x=total_bal_ex_mort))+
  geom_histogram(bins=100,
                 fill="blue")+
  labs(title = "Histograma total_bal_ex_mort",
       subtitle = "Lending Club",
       y="Conteo",
       x="total_bal_ex_mort")+
  theme_bw()

LCST.A %>% 
  ggplot(aes(x=total_bal_ex_mort))+
  geom_density(fill="blue")

LCST.A %>% 
  ggplot(aes(x=total_bal_ex_mort))+
  geom_boxplot(fill="blue")

# Feature 9 

LCST.A %>% 
  ggplot(aes(x=total_bc_limit))+
  geom_histogram(bins=100,
                 fill="blue")+
  labs(title = "Histograma total_bc_limit",
       subtitle = "Lending Club",
       y="Conteo",
       x="total_bc_limit")+
  theme_bw()

LCST.A %>% 
  ggplot(aes(x=total_bc_limit))+
  geom_density(fill="blue")

LCST.A %>% 
  ggplot(aes(x=total_bc_limit))+
  geom_boxplot(fill="blue")

# Feature 10 

table(LCST.A$term)

# Target
table(LCST.A$loan_status)

summary(LCST.A)

################################################################################
#                Evaluación y depuran valores extremos (outliers)
################################################################################

LCST.A %>%
  arrange(desc(dti))
LCST.A %>%
  arrange(dti)
LCST.A <- LCST.A %>% 
  filter(dti >= 0)

LCST.A %>%
  arrange(desc(tot_hi_cred_lim))

LCST.A %>%
  arrange(desc(total_acc))

LCST.A %>%
  arrange(desc(total_bal_ex_mort))

LCST.A %>%
  arrange(desc(total_bc_limit))

LCST.A <- LCST.A %>% 
  filter(total_bc_limit <= 1000000)

################################################################################
#                      Creación de la variable Clase
################################################################################
LCST.A %>%
  select(loan_status) %>%
  table()

LCST.A %>% 
  group_by(loan_status) %>% 
  summarise(Cantidad=n(),
            Prop=n()/nrow(LCST.A))


LCST.A <- LCST.A %>% 
  mutate(loan_status = case_when(loan_status=="Fully Paid"~"Paid",
                           TRUE~"Default"))
str(LCST.A)

LCST.A$loan_status <- factor(LCST.A$loan_status, levels = c("Paid", "Default"))
str(LCST.A)                     

LCST.A <- LCST.A %>% 
  mutate(Clase = case_when(loan_status=="Default"~1,
                           TRUE~0))
str(LCST.A) 
################################################################################
#
#            Creación de los datasets de entrenamiento y pruebas               #
#
################################################################################
#Conjunto de datasets 1 (Entrenamiento y pruebas)
################################################################################
index.1 <- sample.split(LCST.A$Clase,
                       SplitRatio = 0.80)

LCST.A.1.Train <- LCST.A %>%
  subset(index.1==TRUE)

LCST.A.1.Test <- LCST.A %>% 
  subset(index.1==FALSE)

LCST.A.1.Train %>% 
  group_by(Clase) %>% 
  summarise(Cantidad=n(),
            Prop=n()/nrow(LCST.A.1.Train))

LCST.A.1.Test %>% 
  group_by(Clase) %>% 
  summarise(Cantidad=n(),
            Prop=n()/nrow(LCST.A.1.Test))
################################################################################
#Conjunto de datasets 2 (Entrenamiento y pruebas)
################################################################################
index.2 <- sample.split(LCST.A$Clase,
                        SplitRatio = 0.80)

LCST.A.2.Train <- LCST.A %>% 
  subset(index.2==TRUE)

LCST.A.2.Test <- LCST.A %>% 
  subset(index.2==FALSE)

LCST.A.2.Train %>% 
  group_by(Clase) %>% 
  summarise(Cantidad=n(),
            Prop=n()/nrow(LCST.A.2.Train))

LCST.A.2.Test %>% 
  group_by(Clase) %>% 
  summarise(Cantidad=n(),
            Prop=n()/nrow(LCST.A.2.Test))
################################################################################
#Conjunto de datasets 3 (Entrenamiento y pruebas)
################################################################################
index.3 <- sample.split(LCST.A$Clase,
                        SplitRatio = 0.80)

LCST.A.3.Train <- LCST.A %>% 
  subset(index.3==TRUE)

LCST.A.3.Test <- LCST.A %>% 
  subset(index.3==FALSE)

LCST.A.3.Train %>% 
  group_by(Clase) %>% 
  summarise(Cantidad=n(),
            Prop=n()/nrow(LCST.A.3.Train))

LCST.A.3.Test %>% 
  group_by(Clase) %>% 
  summarise(Cantidad=n(),
            Prop=n()/nrow(LCST.A.3.Test))
################################################################################
#Conjunto de datasets 4 (Entrenamiento y pruebas)
################################################################################
index.4 <- sample.split(LCST.A$Clase,
                        SplitRatio = 0.80)

LCST.A.4.Train <- LCST.A %>% 
  subset(index.4==TRUE)

LCST.A.4.Test <- LCST.A %>% 
  subset(index.4==FALSE)

LCST.A.4.Train %>% 
  group_by(Clase) %>% 
  summarise(Cantidad=n(),
            Prop=n()/nrow(LCST.A.4.Train))

LCST.A.4.Test %>% 
  group_by(Clase) %>% 
  summarise(Cantidad=n(),
            Prop=n()/nrow(LCST.A.4.Test))
################################################################################
#Conjunto de datasets 5 (Entrenamiento y pruebas)
################################################################################
index.5 <- sample.split(LCST.A$Clase,
                        SplitRatio = 0.80)

LCST.A.5.Train <- LCST.A %>% 
  subset(index.5==TRUE)

LCST.A.5.Test <- LCST.A %>% 
  subset(index.5==FALSE)

LCST.A.5.Train %>% 
  group_by(Clase) %>% 
  summarise(Cantidad=n(),
            Prop=n()/nrow(LCST.A.5.Train))

LCST.A.5.Test %>% 
  group_by(Clase) %>% 
  summarise(Cantidad=n(),
            Prop=n()/nrow(LCST.A.5.Test))


################################################################################
#
#                              Regresión logística
#
################################################################################
#      Entrenamiento Prueba y evaluación de modelos sobre el dataset 1         #
################################################################################
LR.A.1.1 <- glm(Clase~loan_amnt+int_rate+dti+fico_range_low+fico_range_high+
                tot_hi_cred_lim+total_acc+total_bal_ex_mort+total_bc_limit,
                    data = LCST.A.1.Train,
                    family = binomial)
# El moledo 1 queda descartado debido a que tres de los features no son 
# estadisticamente representativos.
summary(LR.A.1.1)
################################################################################
# Ajuste del modelo 1 en el dataset 1
################################################################################
LR.A.1.2 <- glm(Clase~loan_amnt+int_rate+dti+tot_hi_cred_lim+
                  total_bal_ex_mort+total_bc_limit,
                data = LCST.A.1.Train,
                family = binomial)

summary(LR.A.1.2)

LR.Adj.A.1.2 <- predict(LR.A.1.2,
                  newdata = LCST.A.1.Test,
                  type = "response")

LCST.A.1.2.Test <- LCST.A.1.Test %>% 
  mutate(Probability=LR.Adj.A.1.2,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.1.2.Test$Prediction <- factor(LCST.A.1.2.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.1.2.Test$loan_status, y=LCST.A.1.2.Test$Prediction, prop.chisq=FALSE)

CM.A.1.2 <- LCST.A.1.2.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")
CM.A.1.2
################################################################################
# Ajuste del modelo 2 en el dataset 1
################################################################################
LR.A.1.3 <- glm(Clase~loan_amnt+int_rate+tot_hi_cred_lim+
                  total_bal_ex_mort,
                data = LCST.A.1.Train,
                family = binomial)

summary(LR.A.1.3)

LR.Adj.A.1.3 <- predict(LR.A.1.3,
                        newdata = LCST.A.1.Test,
                        type = "response")

LCST.A.1.3.Test <- LCST.A.1.Test %>% 
  mutate(Probability=LR.Adj.A.1.3,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.1.3.Test$Prediction <- factor(LCST.A.1.3.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.1.3.Test$loan_status, y=LCST.A.1.3.Test$Prediction, prop.chisq=FALSE)

CM.A.1.3 <- LCST.A.1.3.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")


################################################################################
# Ajuste del modelo 3 en el dataset 1
################################################################################
LR.A.1.4 <- glm(Clase~loan_amnt+int_rate+
                  tot_hi_cred_lim+total_acc+total_bal_ex_mort,
                data = LCST.A.1.Train,
                family = binomial)

summary(LR.A.1.4)

LR.Adj.A.1.4 <- predict(LR.A.1.4,
                        newdata = LCST.A.1.Test,
                        type = "response")

LCST.A.1.4.Test <- LCST.A.1.Test %>% 
  mutate(Probability=LR.Adj.A.1.4,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.1.4.Test$Prediction <- factor(LCST.A.1.4.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.1.4.Test$loan_status, y=LCST.A.1.4.Test$Prediction, prop.chisq=FALSE)

CM.A.1.4 <- LCST.A.1.4.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 4 en el dataset 1
################################################################################
LR.A.1.5 <- glm(Clase~loan_amnt+int_rate,
                data = LCST.A.1.Train,
                family = binomial)

summary(LR.A.1.5)

LR.Adj.A.1.5 <- predict(LR.A.1.5,
                        newdata = LCST.A.1.Test,
                        type = "response")

LCST.A.1.5.Test <- LCST.A.1.Test %>% 
  mutate(Probability=LR.Adj.A.1.5,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.1.5.Test$Prediction <- factor(LCST.A.1.5.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.1.5.Test$loan_status, y=LCST.A.1.5.Test$Prediction, prop.chisq=FALSE)

CM.A.1.5 <- LCST.A.1.5.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 5 en el dataset 1
################################################################################
LR.A.1.6 <- glm(Clase~int_rate,
                data = LCST.A.1.Train,
                family = binomial)

summary(LR.A.1.6)

LR.Adj.A.1.6 <- predict(LR.A.1.6,
                        newdata = LCST.A.1.Test,
                        type = "response")

LCST.A.1.6.Test <- LCST.A.1.Test %>% 
  mutate(Probability=LR.Adj.A.1.6,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.1.6.Test$Prediction <- factor(LCST.A.1.6.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.1.6.Test$loan_status, y=LCST.A.1.6.Test$Prediction, prop.chisq=FALSE)

CM.A.1.6 <- LCST.A.1.6.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")


################################################################################
#      Entrenamiento Prueba y evaluación de modelos sobre el dataset 2         #
################################################################################

################################################################################
# Ajuste del modelo 1 en el dataset 2
################################################################################

LR.A.2.2 <- glm(Clase~loan_amnt+int_rate+dti+tot_hi_cred_lim+
                  total_bal_ex_mort+total_bc_limit,
                data = LCST.A.2.Train,
                family = binomial)


summary(LR.A.2.2)

LR.Adj.A.2.2 <- predict(LR.A.2.2,
                        newdata = LCST.A.2.Test,
                        type = "response")

LCST.A.2.2.Test <- LCST.A.2.Test %>% 
  mutate(Probability=LR.Adj.A.2.2,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.2.2.Test$Prediction <- factor(LCST.A.2.2.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.2.2.Test$loan_status, y=LCST.A.2.2.Test$Prediction, prop.chisq=FALSE)

CM.A.2.2 <- LCST.A.2.2.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 2 en el dataset 2
################################################################################

LR.A.2.3 <- glm(Clase~loan_amnt+int_rate+tot_hi_cred_lim+
                  total_bal_ex_mort,
                data = LCST.A.2.Train,
                family = binomial)

summary(LR.A.2.3)

LR.Adj.A.2.3 <- predict(LR.A.2.3,
                        newdata = LCST.A.2.Test,
                        type = "response")

LCST.A.2.3.Test <- LCST.A.2.Test %>% 
  mutate(Probability=LR.Adj.A.2.3,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.2.3.Test$Prediction <- factor(LCST.A.2.3.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.2.3.Test$loan_status, y=LCST.A.2.3.Test$Prediction, prop.chisq=FALSE)

CM.A.2.3 <- LCST.A.2.3.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")


################################################################################
# Ajuste del modelo 3 en el dataset 2
################################################################################

LR.A.2.4 <- glm(Clase~loan_amnt+int_rate+
                  tot_hi_cred_lim+total_acc+total_bal_ex_mort,
                data = LCST.A.2.Train,
                family = binomial)

summary(LR.A.2.4)

LR.Adj.A.2.4 <- predict(LR.A.2.4,
                        newdata = LCST.A.2.Test,
                        type = "response")

LCST.A.2.4.Test <- LCST.A.2.Test %>% 
  mutate(Probability=LR.Adj.A.2.4,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.2.4.Test$Prediction <- factor(LCST.A.2.4.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.2.4.Test$loan_status, y=LCST.A.2.4.Test$Prediction, prop.chisq=FALSE)

CM.A.2.4 <- LCST.A.2.4.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 4 en el dataset 2
################################################################################

LR.A.2.5 <- glm(Clase~loan_amnt+int_rate,
                data = LCST.A.2.Train,
                family = binomial)

summary(LR.A.2.5)

LR.Adj.A.2.5 <- predict(LR.A.2.5,
                        newdata = LCST.A.2.Test,
                        type = "response")

LCST.A.2.5.Test <- LCST.A.2.Test %>% 
  mutate(Probability=LR.Adj.A.2.5,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.2.5.Test$Prediction <- factor(LCST.A.2.5.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.2.5.Test$loan_status, y=LCST.A.2.5.Test$Prediction, prop.chisq=FALSE)

CM.A.2.5 <- LCST.A.2.5.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 5 en el dataset 2
################################################################################

LR.A.2.6 <- glm(Clase~int_rate,
                data = LCST.A.2.Train,
                family = binomial)

summary(LR.A.2.6)

LR.Adj.A.2.6 <- predict(LR.A.2.6,
                        newdata = LCST.A.2.Test,
                        type = "response")

LCST.A.2.6.Test <- LCST.A.2.Test %>% 
  mutate(Probability=LR.Adj.A.2.6,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.2.6.Test$Prediction <- factor(LCST.A.2.6.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.2.6.Test$loan_status, y=LCST.A.2.6.Test$Prediction, prop.chisq=FALSE)

CM.A.2.6 <- LCST.A.2.6.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
#      Entrenamiento Prueba y evaluación de modelos sobre el dataset 3         #
################################################################################

################################################################################
# Ajuste del modelo 1 en el dataset 3
################################################################################

LR.A.3.2 <- glm(Clase~loan_amnt+int_rate+dti+tot_hi_cred_lim+
                  total_bal_ex_mort+total_bc_limit,
                data = LCST.A.3.Train,
                family = binomial)

summary(LR.A.3.2)

LR.Adj.A.3.2 <- predict(LR.A.3.2,
                        newdata = LCST.A.3.Test,
                        type = "response")

LCST.A.3.2.Test <- LCST.A.3.Test %>% 
  mutate(Probability=LR.Adj.A.3.2,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.3.2.Test$Prediction <- factor(LCST.A.3.2.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.3.2.Test$loan_status, y=LCST.A.3.2.Test$Prediction, prop.chisq=FALSE)

CM.A.3.2 <- LCST.A.3.2.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 2 en el dataset 3
################################################################################

LR.A.3.3 <- glm(Clase~loan_amnt+int_rate+tot_hi_cred_lim+
                  total_bal_ex_mort,
                data = LCST.A.3.Train,
                family = binomial)

summary(LR.A.3.3)

LR.Adj.A.3.3 <- predict(LR.A.3.3,
                        newdata = LCST.A.3.Test,
                        type = "response")

LCST.A.3.3.Test <- LCST.A.3.Test %>% 
  mutate(Probability=LR.Adj.A.3.3,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.3.3.Test$Prediction <- factor(LCST.A.3.3.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.3.3.Test$loan_status, y=LCST.A.3.3.Test$Prediction, prop.chisq=FALSE)

CM.A.3.3 <- LCST.A.3.3.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")


################################################################################
# Ajuste del modelo 3 en el dataset 3
################################################################################

LR.A.3.4 <- glm(Clase~loan_amnt+int_rate+
                  tot_hi_cred_lim+total_acc+total_bal_ex_mort,
                data = LCST.A.3.Train,
                family = binomial)

summary(LR.A.3.4)

LR.Adj.A.3.4 <- predict(LR.A.3.4,
                        newdata = LCST.A.3.Test,
                        type = "response")

LCST.A.3.4.Test <- LCST.A.3.Test %>% 
  mutate(Probability=LR.Adj.A.3.4,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.3.4.Test$Prediction <- factor(LCST.A.3.4.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.3.4.Test$loan_status, y=LCST.A.3.4.Test$Prediction, prop.chisq=FALSE)

CM.A.3.4 <- LCST.A.3.4.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 4 en el dataset 3
################################################################################

LR.A.3.5 <- glm(Clase~loan_amnt+int_rate,
                data = LCST.A.3.Train,
                family = binomial)

summary(LR.A.3.5)

LR.Adj.A.3.5 <- predict(LR.A.3.5,
                        newdata = LCST.A.3.Test,
                        type = "response")

LCST.A.3.5.Test <- LCST.A.3.Test %>% 
  mutate(Probability=LR.Adj.A.3.5,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.3.5.Test$Prediction <- factor(LCST.A.3.5.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.3.5.Test$loan_status, y=LCST.A.3.5.Test$Prediction, prop.chisq=FALSE)

CM.A.3.5 <- LCST.A.3.5.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 5 en el dataset 3
################################################################################

LR.A.3.6 <- glm(Clase~int_rate,
                data = LCST.A.3.Train,
                family = binomial)

summary(LR.A.3.6)

LR.Adj.A.3.6 <- predict(LR.A.3.6,
                        newdata = LCST.A.3.Test,
                        type = "response")

LCST.A.3.6.Test <- LCST.A.3.Test %>% 
  mutate(Probability=LR.Adj.A.3.6,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.3.6.Test$Prediction <- factor(LCST.A.3.6.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.3.6.Test$loan_status, y=LCST.A.3.6.Test$Prediction, prop.chisq=FALSE)

CM.A.3.6 <- LCST.A.3.6.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
#      Entrenamiento Prueba y evaluación de modelos sobre el dataset 4         #
################################################################################

################################################################################
# Ajuste del modelo 1 en el dataset 4
################################################################################


LR.A.4.2 <- glm(Clase~loan_amnt+int_rate+dti+tot_hi_cred_lim+
                  total_bal_ex_mort+total_bc_limit,
                data = LCST.A.4.Train,
                family = binomial)

summary(LR.A.4.2)

LR.Adj.A.4.2 <- predict(LR.A.4.2,
                        newdata = LCST.A.4.Test,
                        type = "response")

LCST.A.4.2.Test <- LCST.A.4.Test %>% 
  mutate(Probability=LR.Adj.A.4.2,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.4.2.Test$Prediction <- factor(LCST.A.4.2.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.4.2.Test$loan_status, y=LCST.A.4.2.Test$Prediction, prop.chisq=FALSE)

CM.A.4.2 <- LCST.A.4.2.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 2 en el dataset 4
################################################################################
LR.A.4.3 <- glm(Clase~loan_amnt+int_rate+tot_hi_cred_lim+
                  total_bal_ex_mort,
                data = LCST.A.4.Train,
                family = binomial)

summary(LR.A.4.3)

LR.Adj.A.4.3 <- predict(LR.A.4.3,
                        newdata = LCST.A.4.Test,
                        type = "response")

LCST.A.4.3.Test <- LCST.A.4.Test %>% 
  mutate(Probability=LR.Adj.A.4.3,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.4.3.Test$Prediction <- factor(LCST.A.4.3.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.4.3.Test$loan_status, y=LCST.A.4.3.Test$Prediction, prop.chisq=FALSE)

CM.A.4.3 <- LCST.A.4.3.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")


################################################################################
# Ajuste del modelo 3 en el dataset 4
################################################################################
LR.A.4.4 <- glm(Clase~loan_amnt+int_rate+
                  tot_hi_cred_lim+total_acc+total_bal_ex_mort,
                data = LCST.A.4.Train,
                family = binomial)

summary(LR.A.4.4)

LR.Adj.A.4.4 <- predict(LR.A.4.4,
                        newdata = LCST.A.4.Test,
                        type = "response")

LCST.A.4.4.Test <- LCST.A.4.Test %>% 
  mutate(Probability=LR.Adj.A.4.4,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.4.4.Test$Prediction <- factor(LCST.A.4.4.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.4.4.Test$loan_status, y=LCST.A.4.4.Test$Prediction, prop.chisq=FALSE)

CM.A.4.4 <- LCST.A.4.4.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 4 en el dataset 4
################################################################################
LR.A.4.5 <- glm(Clase~loan_amnt+int_rate,
                data = LCST.A.4.Train,
                family = binomial)

summary(LR.A.4.5)

LR.Adj.A.4.5 <- predict(LR.A.4.5,
                        newdata = LCST.A.4.Test,
                        type = "response")

LCST.A.4.5.Test <- LCST.A.4.Test %>% 
  mutate(Probability=LR.Adj.A.4.5,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.4.5.Test$Prediction <- factor(LCST.A.4.5.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.4.5.Test$loan_status, y=LCST.A.4.5.Test$Prediction, prop.chisq=FALSE)

CM.A.4.5 <- LCST.A.4.5.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 5 en el dataset 4
################################################################################
LR.A.4.6 <- glm(Clase~int_rate,
                data = LCST.A.4.Train,
                family = binomial)

summary(LR.A.4.6)

LR.Adj.A.4.6 <- predict(LR.A.4.6,
                        newdata = LCST.A.4.Test,
                        type = "response")

LCST.A.4.6.Test <- LCST.A.4.Test %>% 
  mutate(Probability=LR.Adj.A.4.6,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.4.6.Test$Prediction <- factor(LCST.A.4.6.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.4.6.Test$loan_status, y=LCST.A.4.6.Test$Prediction, prop.chisq=FALSE)

CM.A.4.6 <- LCST.A.4.6.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")


################################################################################
#      Entrenamiento Prueba y evaluación de modelos sobre el dataset 5         #
################################################################################

################################################################################
# Ajuste del modelo 1 en el dataset 5
################################################################################

LR.A.5.2 <- glm(Clase~loan_amnt+int_rate+dti+tot_hi_cred_lim+
                  total_bal_ex_mort+total_bc_limit,
                data = LCST.A.5.Train,
                family = binomial)

summary(LR.A.5.2)

LR.Adj.A.5.2 <- predict(LR.A.5.2,
                        newdata = LCST.A.5.Test,
                        type = "response")

LCST.A.5.2.Test <- LCST.A.5.Test %>% 
  mutate(Probability=LR.Adj.A.5.2,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.5.2.Test$Prediction <- factor(LCST.A.5.2.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.5.2.Test$loan_status, y=LCST.A.5.2.Test$Prediction, prop.chisq=FALSE)

CM.A.5.2 <- LCST.A.5.2.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 2 en el dataset 5
################################################################################
LR.A.5.3 <- glm(Clase~loan_amnt+int_rate+tot_hi_cred_lim+
                  total_bal_ex_mort,
                data = LCST.A.5.Train,
                family = binomial)

summary(LR.A.5.3)

LR.Adj.A.5.3 <- predict(LR.A.5.3,
                        newdata = LCST.A.5.Test,
                        type = "response")

LCST.A.5.3.Test <- LCST.A.5.Test %>% 
  mutate(Probability=LR.Adj.A.5.3,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.5.3.Test$Prediction <- factor(LCST.A.5.3.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.5.3.Test$loan_status, y=LCST.A.5.3.Test$Prediction, prop.chisq=FALSE)

CM.A.5.3 <- LCST.A.5.3.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")


################################################################################
# Ajuste del modelo 3 en el dataset 5
################################################################################
LR.A.5.4 <- glm(Clase~loan_amnt+int_rate+
                  tot_hi_cred_lim+total_acc+total_bal_ex_mort,
                data = LCST.A.5.Train,
                family = binomial)

summary(LR.A.5.4)

LR.Adj.A.5.4 <- predict(LR.A.5.4,
                        newdata = LCST.A.5.Test,
                        type = "response")

LCST.A.5.4.Test <- LCST.A.5.Test %>% 
  mutate(Probability=LR.Adj.A.5.4,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.5.4.Test$Prediction <- factor(LCST.A.5.4.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.5.4.Test$loan_status, y=LCST.A.5.4.Test$Prediction, prop.chisq=FALSE)

CM.A.5.4 <- LCST.A.5.4.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 4 en el dataset 5
################################################################################
LR.A.5.5 <- glm(Clase~loan_amnt+int_rate,
                data = LCST.A.5.Train,
                family = binomial)

summary(LR.A.5.5)

LR.Adj.A.5.5 <- predict(LR.A.5.5,
                        newdata = LCST.A.5.Test,
                        type = "response")

LCST.A.5.5.Test <- LCST.A.5.Test %>% 
  mutate(Probability=LR.Adj.A.5.5,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.5.5.Test$Prediction <- factor(LCST.A.5.5.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.5.5.Test$loan_status, y=LCST.A.5.5.Test$Prediction, prop.chisq=FALSE)

CM.A.5.5 <- LCST.A.5.5.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
# Ajuste del modelo 5 en el dataset 5
################################################################################
LR.A.5.6 <- glm(Clase~int_rate,
                data = LCST.A.5.Train,
                family = binomial)

summary(LR.A.5.6)

LR.Adj.A.5.6 <- predict(LR.A.5.6,
                        newdata = LCST.A.5.Test,
                        type = "response")

LCST.A.5.6.Test <- LCST.A.5.Test %>% 
  mutate(Probability=LR.Adj.A.5.6,
         Prediction=case_when(Probability>=0.5~"Default",
                              TRUE~"Paid"))

LCST.A.5.6.Test$Prediction <- factor(LCST.A.5.6.Test$Prediction, levels = c("Paid", "Default"))

CrossTable(x=LCST.A.5.6.Test$loan_status, y=LCST.A.5.6.Test$Prediction, prop.chisq=FALSE)

CM.A.5.6 <- LCST.A.5.6.Test %>% 
  select(loan_status,Prediction) %>% 
  table() %>% 
  confusionMatrix(reference="Default")

################################################################################
#           Estimación de matrices de confusión promedio
################################################################################
Prom.Mod.1 <- matrix(0,2,2)
Prom.Mod.2 <- matrix(0,2,2)
Prom.Mod.3 <- matrix(0,2,2)
Prom.Mod.4 <- matrix(0,2,2)
Prom.Mod.5 <- matrix(0,2,2)

for(i in 1:2){
  for(j in 1:2){
    Prom.Mod.1[i,j] <- CM.A.1.2$table[i,j]+CM.A.2.2$table[i,j]+CM.A.3.2$table[i,j]+CM.A.4.2$table[i,j]+CM.A.5.2$table[i,j]
    Prom.Mod.2[i,j] <- CM.A.1.3$table[i,j]+CM.A.2.3$table[i,j]+CM.A.3.3$table[i,j]+CM.A.4.3$table[i,j]+CM.A.5.3$table[i,j]
    Prom.Mod.3[i,j] <- CM.A.1.4$table[i,j]+CM.A.2.4$table[i,j]+CM.A.3.4$table[i,j]+CM.A.4.4$table[i,j]+CM.A.5.4$table[i,j]
    Prom.Mod.4[i,j] <- CM.A.1.5$table[i,j]+CM.A.2.5$table[i,j]+CM.A.3.5$table[i,j]+CM.A.4.5$table[i,j]+CM.A.5.5$table[i,j]
    Prom.Mod.5[i,j] <- CM.A.1.6$table[i,j]+CM.A.2.6$table[i,j]+CM.A.3.6$table[i,j]+CM.A.4.6$table[i,j]+CM.A.5.6$table[i,j]
  }
}

for(i in 1:2){
  for(j in 1:2){
    Prom.Mod.1[i,j] <- Prom.Mod.1[i,j]/5
    Prom.Mod.2[i,j] <- Prom.Mod.2[i,j]/5
    Prom.Mod.3[i,j] <- Prom.Mod.3[i,j]/5
    Prom.Mod.4[i,j] <- Prom.Mod.4[i,j]/5
    Prom.Mod.5[i,j] <- Prom.Mod.5[i,j]/5
  }
}

Prom.Mod.1
Prom.Mod.2
Prom.Mod.3
Prom.Mod.4
Prom.Mod.5








