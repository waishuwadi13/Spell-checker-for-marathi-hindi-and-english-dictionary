import numpy as np
import os

def load_words(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
         return [line.strip().lower() for line in f if line.strip()]

def build_index(words):
    n = len(words)
    counts = np.zeros((n,26),dtype = np.uint8)
    lengths = np.zeros(n,dtype = np.uint16)
    for i,w in enumerate(words):
        lengths[i] = len(w)
        for letters in w:
            if 'a' <= letters <= 'z':
                counts[i,ord(letters) - 97] += 1
    return counts,lengths

def save_index(words,counts,lengths,path="spell_index.npz"):
    np.savez_compressed(path,words=np.array(words,dtype=object),counts=counts,lengths=lengths)

def load_index(path="spell_index.npz"):
    data = np.load(path,allow_pickle=True)
    return data["words"].tolist(),data["counts"],data["lengths"]

def match(w1_set,w2_set,max_dist):
    max_len = max(len(w1_set),len(w2_set))
    common = len(w1_set & w2_set)
    return common >= max_len - max_dist

def edit_distance(w1,w2,max_dist):
    m,n = len(w1),len(w2)
    if abs(m-n) > max_dist:
       return max_dist +1
    prev = np.arange(n+1,dtype=np.uint16)
    curr = np.zeros(n+1,dtype=np.uint16)
    for i in range(1,m+1):
        curr[0] = i
        row_min = i
        for j in range(1,n+1):
            cost = 0 if w1[i-1] == w2[j-1] else 1
            curr[j] = min(prev[j]+1,curr[j-1]+1,prev[j-1]+ cost)
            if curr[j] < row_min:
               row_min = curr[j]
        if row_min > max_dist:
           return max_dist+1
        prev,curr = curr,prev
    return prev[n]

def get_suggestions(query,words,counts,lengths, num=3):
    query = query.lower()
    if len(query) > 50:
       query = query[:50]

    q_counts = np.zeros(26,dtype = np.uint8)
    q_set = set()
    for letters in query:
        if 'a' <= letters <= 'z':
            idx =ord(letters) - 97
            q_counts[idx] += 1
            q_set.add(letters)
    q_len = len(query)

    best = []
    max_edit = max(2,int(q_len * 0.15))
    upper_limit = max(15,q_len // 2+5)

    while len(best) < num and max_edit <= upper_limit:
          len_ok = np.abs(lengths - q_len) <= max_edit
          l1_dists = np.abs(counts.astype(np.int16) - q_counts.astype(np.int16)).sum(axis = 1)
          count_ok = l1_dists <= 2 * max_edit

          mask = len_ok & count_ok
          indices = np.where(mask)[0]

          if len(indices) > 1000:
             scores = l1_dists[indices] + np.abs(lengths[indices] - q_len)
             order = np.argsort(scores)
             indices = indices[order[:1000]]

          seen = {w for _,w in best}
          for idx in indices:
              w = words[idx]
              if w in seen:
                 continue

              w_set = set(w)
              if not match(q_set,w_set,max_edit):
                 continue

              curr_max = best[-1][0] if len(best) >= num else max_edit
              dist = edit_distance(query,w,curr_max)
              if dist <= curr_max:
                 best.append((dist,w))
                 best.sort(key=lambda x:(x[0],x[1]))
                 if len(best) > num:
                    best.pop()
          max_edit += 1 
    
    return [(word,dist) for dist,word in best]

def main():
    index_file = "spell_index.npz"
    if not os.path.exists(index_file):
       words =load_words("english_words_alpha.txt")
       counts,lengths = build_index(words)
       save_index(words,counts,lengths,index_file)
    else: 
       words,counts,lengths = load_index(index_file)

    while True:
       q = input("\nEnter input:").strip()
       if q.lower() == "quit":
          break
       if not q:
          continue

       suggestions = get_suggestions(q,words,counts,lengths,num = 25)

       if suggestions:
            for w, d in suggestions:
                print(f"{w} (distance {d})")
       else:
            print("No suggestions found")

if __name__ == "__main__":
    main()

