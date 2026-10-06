import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import load_sample_image

from kmeans import kmeans, best_of, assign

st.set_page_config(page_title="K-means Studio", layout="wide")
st.title("K-means Studio")
st.caption("One algorithm, many views. K-means is written from scratch in kmeans.py. "
           "Each pixel is a point in (R, G, B) space and k-means finds the k colours that describe the image best.")

# ---------- load image ----------
up = st.sidebar.file_uploader("Your image (optional)", type=["jpg", "jpeg", "png"])
if up:
    img = Image.open(up).convert("RGB")
else:
    which = st.sidebar.selectbox("Sample image", ["china.jpg", "flower.jpg"])
    img = Image.fromarray(load_sample_image(which))
img.thumbnail((360, 360))
pixels = np.asarray(img, dtype=float) / 255.0
h, w, _ = pixels.shape
P = pixels.reshape(-1, 3)
k = st.sidebar.slider("k (number of colours)", 2, 32, 8)
rng = np.random.default_rng(0)
sample = P[rng.choice(len(P), min(4000, len(P)), replace=False)]   # train on a sample, then paint every pixel


@st.cache_data(show_spinner=False)
def fit(k_, n_init=5):
    c, _, inertia, _ = best_of(sample, k_, n_init=n_init)
    return c, inertia


centers, _ = fit(k)
labels, _ = assign(P, centers)
out = centers[labels].reshape(h, w, 3)

tabs = st.tabs(["1. Compress", "2. Watch it learn", "3. Palette", "4. Why the start matters", "5. Choosing k"])

# ---------- 1 compress ----------
with tabs[0]:
    c1, c2 = st.columns(2)
    c1.image(pixels, caption="Original", width="stretch")
    c2.image(np.clip(out, 0, 1), caption=f"Only {k} colours", width="stretch")
    mse = ((pixels - out) ** 2).mean()
    psnr = 10 * np.log10(1 / mse)
    bits = max(1, int(np.ceil(np.log2(k))))
    orig_bytes = h * w * 3
    new_bytes = h * w * bits / 8 + k * 3
    m1, m2, m3 = st.columns(3)
    m1.metric("Bits per pixel", f"{bits} (was 24)")
    m2.metric("Size", f"{new_bytes / 1024:.1f} KB", f"{100 * (1 - new_bytes / orig_bytes):.0f}% smaller")
    m3.metric("Quality (PSNR, higher better)", f"{psnr:.1f} dB")
    st.write("How it works: instead of storing 3 numbers per pixel, store a **palette** of k colours "
             "once, then for every pixel store just the **index** of its palette colour.")

# ---------- 2 watch it learn ----------
with tabs[1]:
    st.write("Pixels are floating in colour space. Centres (big diamonds) start somewhere and chase the crowds.")
    init = st.radio("Starting method", ["random", "kmeans++"], horizontal=True)
    seed = st.number_input("Random seed", 0, 99, 1)
    _, _, _, hist = kmeans(sample, k, init, seed=int(seed))
    step = st.slider("Iteration", 1, len(hist), 1)
    cen, lab, inert = hist[step - 1]
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection="3d")
        show = sample[:1200]
        ax.scatter(show[:, 0], show[:, 1], show[:, 2], c=show, s=6, alpha=0.5)
        ax.scatter(cen[:, 0], cen[:, 1], cen[:, 2], c=np.clip(cen, 0, 1), s=260, marker="D", edgecolors="black")
        ax.set_xlabel("Red"); ax.set_ylabel("Green"); ax.set_zlabel("Blue")
        ax.set_title(f"Iteration {step} of {len(hist)}   inertia {inert:,.0f}")
        st.pyplot(fig)
    with c2:
        lab_all, _ = assign(P, cen)
        st.image(np.clip(cen[lab_all].reshape(h, w, 3), 0, 1), caption=f"Image painted with iteration {step} colours",
                 width="stretch")
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.plot(range(1, len(hist) + 1), [x[2] for x in hist], marker="o"); ax.axvline(step, color="red", ls="--")
    ax.set_xlabel("iteration"); ax.set_ylabel("inertia"); ax.set_title("Total error only ever goes down")
    st.pyplot(fig)

# ---------- 3 palette ----------
with tabs[2]:
    share = np.bincount(labels, minlength=k) / len(labels)
    order = np.argsort(share)[::-1]
    fig, ax = plt.subplots(figsize=(10, 1.6))
    left = 0
    for j in order:
        ax.barh(0, share[j], left=left, color=np.clip(centers[j], 0, 1)); left += share[j]
    ax.axis("off"); ax.set_title("Palette, wider = more of the image")
    st.pyplot(fig)
    cols = st.columns(min(k, 8))
    for i, j in enumerate(order):
        hexv = "#%02x%02x%02x" % tuple((np.clip(centers[j], 0, 1) * 255).astype(int))
        with cols[i % len(cols)]:
            st.markdown(f"<div style='background:{hexv};height:46px;border-radius:6px'></div>", unsafe_allow_html=True)
            st.caption(f"{hexv}  \n{share[j] * 100:.1f}%")

# ---------- 4 initialisation ----------
with tabs[3]:
    st.write("K-means can end in different places depending on where it starts. "
             "We run it 30 times with each starting method and compare the final error.")
    if st.button("Run experiment"):
        r = [kmeans(sample, k, "random", seed=s)[2] for s in range(30)]
        pp = [kmeans(sample, k, "kmeans++", seed=s)[2] for s in range(30)]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(r, alpha=0.6, label=f"random (avg {np.mean(r):.0f})", bins=12)
        ax.hist(pp, alpha=0.6, label=f"k-means++ (avg {np.mean(pp):.0f})", bins=12)
        ax.set_xlabel("final inertia (lower is better)"); ax.set_ylabel("runs"); ax.legend()
        st.pyplot(fig)
        st.write(f"Best of 30: random {min(r):.0f}, k-means++ {min(pp):.0f}. "
                 f"Worst of 30: random {max(r):.0f}, k-means++ {max(pp):.0f}.")

# ---------- 5 choosing k ----------
with tabs[4]:
    ks = list(range(1, 17))
    inert = [fit(kk, 3)[1] for kk in ks]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(ks, inert, marker="o"); ax.axvline(k, ls="--", color="grey")
    ax.set_xlabel("k"); ax.set_ylabel("inertia"); ax.set_title("Elbow plot")
    st.pyplot(fig)
    st.write("More colours always lowers the error, but after the bend each extra colour barely helps. "
             "Pick k near the bend.")
