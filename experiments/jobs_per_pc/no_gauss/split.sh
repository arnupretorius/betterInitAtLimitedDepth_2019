COUNT=1
while read line;
do
  echo "$line" > "experiment_$COUNT.txt"
  ((COUNT += 1))
done < all.txt