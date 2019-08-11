COUNT=1
while read line;
do
  echo "$line" > "hyp_$COUNT.txt"
  ((COUNT += 1))
done < all.txt