######################################################################################################################################
#                                           TEXTURE ANALYSIS (GLCM)
#
# 1st parameter: location of teh graph.xml file
# 2nd parameter: location of the source product
# 3rd parameter: location of the output folder
#
#                                           HOW TO RUN THE SCRIPT
# bash <location of the script.sh> <location of the graph.xml> <location of source_data> <location of the output folder>
#
#                                               EXAMPLE
#                       bash texture.sh graphs/texture.xml input output/
######################################################################################################################################

gptPath=~/snap/bin/gpt
graph="$1"
source_folder="$2"
target_product="$3"
repo_folder="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


function get_date () {
    echo $1 | sed -r 's/[^ ]*(.......)T(......){2}[^ ]*\.zip/\2/g'
}

file=($(find "${source_folder}"/*.zip ))
source_product="${file}"
source_product_date=$(get_date "${source_product}")
graph="${repo_folder}/graphs/texture.xml"
${gptPath} "${graph}" \
-PsourceProduct="${source_product}" \
-PtargetProduct="${target_product}/GLCM_$source_product_date"

