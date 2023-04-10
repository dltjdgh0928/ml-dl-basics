import re

# 엑손 추출을 위한 정규식 패턴
exon_pattern = re.compile(r"TAC(?:.{3})*?(?:ATC|ACT|ATT)")

# DNA 서열
dna_sequence = "TACTACATCATCTACATC"

# 엑손 추출
exons = []
last_stop = 0  # 이전 종결코돈의 위치
for match in exon_pattern.finditer(dna_sequence):
    # 개시코돈의 위치
    start = match.start()
    # 종결코돈의 위치
    end = match.end()
    # 엑손의 시작 위치가 이전 종결코돈 위치보다 뒤에 있으면 중복 엑손
    if start % 3 == 0 and start >= last_stop:
        exons.append(match.group())
        last_stop = end

# 결과 출력
if exons:
    print("Exons found:")
    for exon in exons:
        print(exon)
    print("Number of exons:", len(exons))
else:
    print("No exons found")
