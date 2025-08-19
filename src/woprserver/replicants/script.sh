# From the root of your project
find . -type f ! -name "all_files.log" | while read -r file; do
  if file "$file" | grep -q "text"; then
    echo "==================== $file ====================" >> all_files.log
    cat "$file" >> all_files.log
    echo -e "\n\n" >> all_files.log
  else
    echo "Skipping binary file: $file"
  fi
done
