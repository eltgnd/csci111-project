import pickle


with open("progress.pkl", "rb") as f:
    progress = pickle.load(f)


models = {'logreg':progress['logreg'], 'rfc':progress['rfc']}
df = progress['x_train']


with open('models.pkl', 'wb') as f:
    pickle.dump(models, f)

with open('xtrain_df.pkl', 'wb') as f:
    pickle.dump(df, f)
