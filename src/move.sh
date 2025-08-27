BUCKET="gs://10bt_gpt2"
DEST="$BUCKET/train"

for i in $(seq -w 000004 000080); do
    FILE="edufineweb_train_${i}.npy"
    echo "Moving $FILE ..."
    gsutil mv "$BUCKET/$FILE" "$DEST/"
done
