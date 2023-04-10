import numpy as np

# 염기서열 데이터
data = "agatattggaactttatactttatttttggagcctgatctggaatagtgggaacttcattaagtatcttaattcgaacagaattaagtcatcctggagcat\
ttattggaagtatagttgaaaatggagctggaactggatgaacagtataccccattatccttctaatttagctcatacaggagcctcagttgatttatcaatttttct\
ttacatttagctggaatttcttctattcttggagctgtaaattttattactacagtaattaatatacgatctacaggaattactttagatcgtatacctttatttgt\
ttgatcagtagtaattactgctattttactactcttatctttacctgtattagcaggagctattactatacttttaactgatcgaattttaacacttcatttttt\
gacccaattggtggaggat"

# 문자열로 된 염기서열을 숫자 배열로 변환
data = data.replace('a', '1').replace('t', '2').replace('c', '3').replace('g', '4').replace('y', '5').replace('w', '6').replace('r', '7').replace('k', '8').replace('v', '9').replace('n', '10').replace('s', '11').replace('m', '12')
data = np.array([int(i) for i in data])

# 개시코돈과 종결코돈
start_codon = np.array([2, 1, 3])
stop_codons = np.array([[1, 2, 3], [1, 3, 2], [1, 2, 2]])

# 엑손 추출을 위한 리스트
exon_list = []

# 데이터에서 엑손 추출
while data.size > 2:
    if np.array_equal(data[:3], start_codon):
        for stop_codon in stop_codons:
            stop_idx = np.where(np.array_equal(stop_codon, data))[0]
            if stop_idx.size > 0 and (stop_idx % 3 == 2) and (stop_idx >= 2):
                stop_idx = stop_idx[0]
                exon = data[:stop_idx + 3]
                if np.array_equal(exon[-3:], stop_codon):
                    exon_list.append(exon)
                    data = data[stop_idx + 3:]
                    break

print(exon_list)
print(exon_list.shape)
