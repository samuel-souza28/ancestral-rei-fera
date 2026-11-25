# Gera caminhada e corrida do Jin por peças a partir do quadro parado (MD Supremo §5.5).
# Ciclo de 4 quadros: 1 contato A (perna da esquerda da imagem à frente, braço da direita à frente),
# 2 passagem, 3 contato B (lados trocados), 4 passagem.
import numpy as np, cv2, sys, json
from PIL import Image

SRC = '/home/claude/ancestral-rei-fera/aide_split/assets/jin_idle_v342.webp'
idle = np.array(Image.open(SRC).convert('RGBA'))
C = 256
NAMES = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

# parâmetros medidos no quadro parado (coordenadas da célula 256x256)
P = {
 'N':  dict(t='front', hem=165, split=128, aL=101, aR=155, elbow=118, hand=174, face=(0, -1)),
 'S':  dict(t='front', hem=168, split=129, aL=103, aR=155, elbow=120, hand=177, face=(0, 1)),
 'NE': dict(t='front', hem=164, split=128, aL=110, aR=154, elbow=120, hand=172, face=(1, -1)),
 'NW': dict(t='front', hem=166, split=134, aL=110, aR=157, elbow=122, hand=174, face=(-1, -1)),
 'SE': dict(t='front', hem=167, split=135, aL=106, aR=158, elbow=122, hand=177, face=(1, 1)),
 'SW': dict(t='front', hem=167, split=136, aL=110, aR=160, elbow=122, hand=174, face=(-1, 1)),
 'E':  dict(t='side', hem=166, hip=(133, 158), arm=(97, 124, 114, 178), pivot=(113, 114), face=(1, 0)),
 'W':  dict(t='side', hem=166, hip=(127, 158), arm=(129, 159, 114, 178), pivot=(142, 114), face=(-1, 0)),
}
FOOT = 244

# amplitudes (px da célula 256; o jogo mostra a ~0,4x)
AMP = {
 'walk': dict(down=6, lift=12, pass_lift=5, shear=6, arm=8, arm_sh=3, side=20, armdeg=30, lean=0, run=False),
 'run':  dict(down=8, lift=18, pass_lift=10, shear=10, arm=12, arm_sh=6, side=30, armdeg=46, lean=3, run=True),
}

def stretch(layer, y0, y1, dy, dx=0):
    """Estica/encolhe as linhas y0..y1 de uma camada: a linha y0 fica parada, a y1 desce dy (e anda dx)."""
    out = np.zeros_like(layer)
    H = layer.shape[0]
    L = y1 - y0
    nb = y1 + dy
    for y in range(max(0, y0), min(H, max(nb, y1) + 1)):
        if y > nb:
            continue
        t = (y - y0) / max(1, (nb - y0))
        sy = int(round(y0 + t * L))
        if sy < 0 or sy >= H:
            continue
        sh = int(round(dx * t))
        row = layer[sy]
        if sh > 0:
            out[y, sh:] = row[:-sh]
        elif sh < 0:
            out[y, :sh] = row[-sh:]
        else:
            out[y] = row
    # parte de cima (acima de y0) fica igual
    out[:y0] = layer[:y0]
    return out

def over(dst, src):
    a = src[..., 3:4].astype(np.float32) / 255
    o = dst.astype(np.float32)
    o[..., :3] = src[..., :3] * a + o[..., :3] * (1 - a)
    o[..., 3:4] = np.maximum(o[..., 3:4], src[..., 3:4])
    return o.astype(np.uint8)

def rot(layer, pivot, deg):
    M = cv2.getRotationMatrix2D((float(pivot[0]), float(pivot[1])), deg, 1.0)
    return cv2.warpAffine(layer, M, (layer.shape[1], layer.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def darker(layer, k=0.72):
    o = layer.copy()
    o[..., :3] = (o[..., :3].astype(np.float32) * k).astype(np.uint8)
    return o

def shift(layer, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(layer, M, (layer.shape[1], layer.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def front_frame(cell, p, A, ph):
    """ph: 0 contato A, 1 passagem (perna esquerda passa), 2 contato B, 3 passagem (perna direita passa)."""
    H, W = cell.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    opaque = cell[..., 3] > 0
    armL = (xx < p['aL']) & (yy >= p['elbow'] - 2) & (yy <= p['hand'] + 6) & opaque
    armR = (xx > p['aR']) & (yy >= p['elbow'] - 2) & (yy <= p['hand'] + 6) & opaque
    # abaixo da barra da túnica, a mão só vai até onde a perna começa (a coxa não anda com o braço)
    lowrows = opaque[p['hand'] + 8:FOOT]
    cols = np.where(lowrows.any(axis=0))[0]
    lmin, lmax = (cols.min(), cols.max()) if len(cols) else (0, W)
    armL &= ~((yy >= p['hem']) & (xx >= lmin))
    armR &= ~((yy >= p['hem']) & (xx <= lmax))
    legs = (yy >= p['hem']) & opaque & ~armL & ~armR
    # divisão das pernas linha a linha: no vão transparente perto do meio (ou no contorno mais escuro), suavizada
    lum = cell[..., :3].astype(np.float32).mean(axis=2)
    cut = np.full(H, p['split'], np.int32)
    for y in range(p['hem'], H):
        lo, hi = p['split'] - 14, p['split'] + 14
        seg = opaque[y, lo:hi]
        best, bl, run, rs = -1, 0, 0, 0
        for i, v in enumerate(seg):
            if not v:
                if run == 0: rs = i
                run += 1
                if run > bl: bl, best = run, rs + run // 2
            else:
                run = 0
        if bl > 0:
            cut[y] = lo + best
        else:
            cut[y] = lo + int(np.argmin(lum[y, lo:hi]))
    from scipy.ndimage import median_filter
    cut[p['hem']:] = median_filter(cut[p['hem']:], size=9, mode='nearest')
    legL = legs & (xx < cut[:, None])
    legR = legs & (xx >= cut[:, None])
    # pedacinhos soltos (pontas de dedo, barra da túnica) não andam com a perna: ficam parados no corpo
    loose = np.zeros_like(legs)
    for a in (legL, legR):
        n, lab = cv2.connectedComponents(a.astype(np.uint8), connectivity=8)
        if n > 2:
            sizes = [(lab == k).sum() for k in range(1, n)]
            keep = 1 + int(np.argmax(sizes))
            for k in range(1, n):
                if k != keep:
                    m = lab == k; a[m] = False; loose |= m
    legs = legs & ~loose
    def part(m):
        o = np.zeros_like(cell); o[m] = cell[m]; return o
    body = part(opaque & ~armL & ~armR & ~legs)
    fx, fy = p['face']
    # pernas: a da frente desce (vem para a câmera quando olha para baixo) e anda para o lado de onde olha
    def leg(layer, state):
        if state == 'fwd':
            dy = A['down'] if fy >= 0 else -A['lift'] // 2
            return stretch(layer, p['hem'], FOOT, dy, fx * A['shear'])
        if state == 'back':
            dy = -A['lift'] if fy >= 0 else A['down'] // 2
            return stretch(layer, p['hem'], FOOT, dy, -fx * A['shear'])
        if state == 'pass':
            return stretch(layer, p['hem'], FOOT, -A['pass_lift'], fx * 1)
        return layer
    def arm(layer, state):
        if state == 'fwd':
            return stretch(layer, p['elbow'], p['hand'] + 6, A['arm'] if fy >= 0 else -A['arm'], fx * A['arm_sh'])
        if state == 'back':
            return stretch(layer, p['elbow'], p['hand'] + 6, -A['arm'] if fy >= 0 else A['arm'], -fx * A['arm_sh'])
        return layer
    if ph == 0:
        sL, sR, aLs, aRs = 'fwd', 'back', 'back', 'fwd'
    elif ph == 2:
        sL, sR, aLs, aRs = 'back', 'fwd', 'fwd', 'back'
    elif ph == 1:
        sL, sR, aLs, aRs = 'pass', 'stand', 'mid', 'mid'
    else:
        sL, sR, aLs, aRs = 'stand', 'pass', 'mid', 'mid'
    L = leg(part(legL), sL); R = leg(part(legR), sR)
    if A['run'] and ph in (1, 3):
        # corrida: na passagem as duas pernas encolhem (o corpo está no ar)
        L = stretch(part(legL), p['hem'], FOOT, -A['pass_lift'] if sL == 'pass' else -A['pass_lift'] // 2, 0)
        R = stretch(part(legR), p['hem'], FOOT, -A['pass_lift'] if sR == 'pass' else -A['pass_lift'] // 2, 0)
    out = np.zeros_like(cell)
    out = over(out, L); out = over(out, R)
    lean = A['lean']
    b = shift(body, fx * lean, 0) if lean else body
    out = over(out, b)
    al = arm(part(armL), aLs); ar = arm(part(armR), aRs)
    if lean:
        al = shift(al, fx * lean, 0); ar = shift(ar, fx * lean, 0)
    out = over(out, al); out = over(out, ar)
    return out

def side_frame(cell, p, A, ph):
    H, W = cell.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    opaque = cell[..., 3] > 0
    x0, x1, y0, y1 = p['arm']
    armm = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1) & opaque
    legs = (yy >= p['hem']) & opaque & ~armm
    def part(m):
        o = np.zeros_like(cell); o[m] = cell[m]; return o
    body = part(opaque & ~armm & ~legs)
    # tronco atrás do braço: preenche o buraco com as cores vizinhas (cv2.inpaint), só dentro do contorno do corpo
    hole = (armm & (xx > x0 + 2) & (xx < x1 - 2)).astype(np.uint8)
    rgb = cv2.cvtColor(body[..., :3], cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(rgb, hole * 255, 3, cv2.INPAINT_TELEA)
    filled = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    # só pinta onde o corpo existe por trás do braço (entre a borda do tronco)
    torso_cols = (opaque & (yy >= y0) & (yy <= y1))
    fillmask = hole.astype(bool) & torso_cols
    # recorta a borda externa: mantém só colunas que têm corpo dos dois lados na mesma linha
    for y in range(y0, y1 + 1):
        row = opaque[y] & ~armm[y]
        xs = np.where(row)[0]
        if len(xs) == 0:
            fillmask[y] = False; continue
        lo, hi = xs.min(), xs.max()
        fillmask[y, :lo] = False; fillmask[y, hi + 1:] = False
    body[fillmask, :3] = filled[fillmask]; body[fillmask, 3] = 255
    fx = p['face'][0]
    legL = part(legs)
    # pernas de lado: cisalhamento a partir da barra da túnica (a bota continua reta no chão, sem buraco no quadril)
    sh = A['side']; lift = A['pass_lift']
    if ph == 0:
        nearL = stretch(legL, p['hem'], FOOT, 0, fx * sh); farL = stretch(legL, p['hem'], FOOT, -2, -fx * sh)
        an, af = -A['armdeg'], A['armdeg']
    elif ph == 2:
        nearL = stretch(legL, p['hem'], FOOT, -2, -fx * sh); farL = stretch(legL, p['hem'], FOOT, 0, fx * sh)
        an, af = A['armdeg'], -A['armdeg']
    elif ph == 1:
        nearL = stretch(legL, p['hem'], FOOT, -(lift // 3), -fx * 3); farL = stretch(legL, p['hem'], FOOT, -lift, fx * 4)
        an, af = 4, -4
    else:
        nearL = stretch(legL, p['hem'], FOOT, -lift, fx * 4); farL = stretch(legL, p['hem'], FOOT, -(lift // 3), -fx * 3)
        an, af = -4, 4
    farL = darker(farL)
    sgn = -fx
    arml = part(armm)
    nearA = rot(arml, p['pivot'], sgn * an)
    farA = darker(rot(arml, p['pivot'], sgn * af), 0.5)
    lean = A['lean']
    out = np.zeros_like(cell)
    out = over(out, shift(farA, fx * lean, 0))
    out = over(out, farL)
    out = over(out, nearL)
    out = over(out, shift(body, fx * lean, 0))
    out = over(out, shift(nearA, fx * lean, 0))
    return out

def build(kind):
    A = AMP[kind]
    sheet = np.zeros((C * 8, C * 4, 4), np.uint8)
    for r, n in enumerate(NAMES):
        cell = idle[r * C:(r + 1) * C, 0:C].copy()
        p = P[n]
        for ph in range(4):
            f = front_frame(cell, p, A, ph) if p['t'] == 'front' else side_frame(cell, p, A, ph)
            sheet[r * C:(r + 1) * C, ph * C:(ph + 1) * C] = f
    return sheet

if __name__ == '__main__':
    for kind in ('walk', 'run'):
        s = build(kind)
        Image.fromarray(s).save(f'jin_{kind}_gen.png')
    print('ok')
