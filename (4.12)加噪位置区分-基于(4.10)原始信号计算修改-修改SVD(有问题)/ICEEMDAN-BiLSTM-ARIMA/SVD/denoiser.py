import numpy as np
from scipy.linalg import svd



class Denoiser(object):
    '''
    一个通过部分循环矩阵的SVD平滑噪声实值数据序列的类。
    -----
    属性:
        mode: str
            运行模式: "layman" 或 "expert"。
            在 "layman" 模式下，代码自动尝试找到最优的去噪序列。
            在 "expert" 模式下，用户对此有完全的控制权。
        s: 1D 浮点数组
            降序排列的奇异值。
        U: 2D 浮点数组
            左奇异向量集合作为列。
        r: int
            从序列构建的部分循环矩阵的近似矩阵的秩。
    '''

    def __init__(self, mode="layman"):
        '''
        类初始化。
        -----
        参数:
            mode: str
                去噪模式。可从 ["layman", "expert"] 中选择。默认为 "layman"。
                "layman" 授予代码自主权，"expert" 允许用户进行实验。
        -----
        抛出:
            ValueError
                如果模式既不是 "layman" 也不是 "expert"。
        '''
        self._method = {"layman": self._denoise_for_layman, "expert": self._denoise_for_expert}
        if mode not in self._method:
            raise ValueError("unknown mode '{:s}'!".format(mode))
        self.mode = mode

    def split_signal(self, sequence, num_segments=6, overlap_count=200):
        total_length = len(sequence)
        segment_lengths = [total_length // num_segments] * num_segments
        remainder = total_length % num_segments

        # Distribute the remainder among the segments
        for i in range(remainder):
            segment_lengths[i] += 1

        segments = []
        start_index = 0
        for length in segment_lengths[:-1]:  # Exclude the last segment for special treatment
            segments.append(sequence[start_index:start_index + length + overlap_count])
            start_index += length

        # Add the last segment without the extra overlap
        segments.append(sequence[start_index:])
        return segments

    def reconstruct_signal(self, segments, overlap_count=200):
        total_length = sum(len(segment) for segment in segments) - (overlap_count * (len(segments) - 1))
        reconstructed_signal = np.zeros(total_length)
        segment_ends = 0

        for i, segment in enumerate(segments):
            segment_length = len(segment)
            # If it's not the first segment, calculate the overlap average
            if i > 0:
                overlap_start = segment_ends - overlap_count
                for j in range(overlap_count):
                    reconstructed_signal[overlap_start + j] = (
                                                                      reconstructed_signal[overlap_start + j] + segment[
                                                                  j]
                                                              ) / 2
                # Update the segment length excluding the overlap to avoid double counting
                segment_length -= overlap_count

            # Place the non-overlapping part of the segment into the reconstructed signal
            reconstructed_signal[segment_ends:segment_ends + segment_length] = segment[overlap_count:]
            segment_ends += segment_length

        return reconstructed_signal

    def _embed(self, x, m):
        '''
        通过循环左移将1D数组嵌入到2D部分循环矩阵中。
        -----
        参数:
            x: 1D 浮点数组
                输入数组。
            m: int
                构建矩阵的行数。
        -----
        返回:
            X: 2D 浮点数组
                构建的部分循环矩阵。
        '''
        x_ext = np.hstack((x, x[:m-1]))
        shape = (m, x.size)
        strides = (x_ext.strides[0], x_ext.strides[0])
        X = np.lib.stride_tricks.as_strided(x_ext, shape, strides)
        return X

    def _reduce(self, A):
        '''
        通过循环反对角线平均将2D矩阵简化为1D数组。
        -----
        参数:
            A: 2D 浮点数组
                输入矩阵。
        -----
        返回:
            a: 1D 浮点数组
                输出数组。
        '''
        m = A.shape[0]
        A_ext = np.hstack((A[:,-m+1:], A))
        strides = (A_ext.strides[0]-A_ext.strides[1], A_ext.strides[1])
        a = np.mean(np.lib.stride_tricks.as_strided(A_ext[:,m-1:], A.shape, strides), axis=0)
        return a

    def _denoise_for_expert(self, sequence, layer, gap, rank):
        '''
        通过对应部分循环矩阵的低秩近似来平滑噪声序列。
        -----
        参数:
            sequence: 1D 浮点数组
                待去噪的数据序列。
            layer: int
                从矩阵中选择的前导行数。
            gap: float
                序列左右端数据水平之间的差距。
                正值表示右侧水平更高。
            rank: int
                近似矩阵的秩。
        -----
        返回:
            denoised: 1D 浮点数组
                去噪后平滑的序列。
        -----
        抛出:
            AssertionError
                如果条件 1 <= rank <= layer <= sequence.size 不能满足。
        '''
        assert 1 <= rank <= layer <= sequence.size
        self.r = rank
        # 待扣除的线性趋势
        trend = np.linspace(0, gap, sequence.size)
        X = self._embed(sequence-trend, layer)
        # 奇异值分解
        self.U, self.s, Vh = svd(X, full_matrices=False, overwrite_a=True, check_finite=False)
        # 低秩近似
        A = self.U[:,:self.r] @ np.diag(self.s[:self.r]) @ Vh[:self.r]
        denoised = self._reduce(A) + trend
        return denoised

    def _cross_validate(self, x, m):
        '''
        检查去趋势序列的边界水平间隙是否在估计的噪声强度内。
        -----
        参数:
            x: 1D 浮点数数组
                输入数组。
            m: int
                构造的矩阵的行数。
        -----
        返回:
            valid: bool
                交叉验证结果。True 表示去趋势过程有效。
        '''
        X = self._embed(x, m)
        self.U, self.s, self._Vh = svd(X, full_matrices=False, overwrite_a=True, check_finite=False)
        # 使用左奇异向量的归一化平均总变差作为指标，搜索噪声组分。
        # 这个过程每10个奇异向量一批运行。
        self.r = 0
        while True:
            U_sub = self.U[:,self.r:self.r+10]
            NMTV = np.mean(np.abs(np.diff(U_sub,axis=0)), axis=0) / (np.amax(U_sub,axis=0) - np.amin(U_sub,axis=0))
            try:
                # 在大多数情况下，10%的阈值可以区分噪声组分
                self.r += np.argwhere(NMTV > .1)[0,0]
                break
            except IndexError:
                self.r += 10
        # 估计噪声强度，r 标记第一个噪声组分
        noise_stdev = np.sqrt(np.sum(self.s[self.r:]**2) / X.size)
        # 估计去趋势后的边界水平间隙
        gap = np.abs(x[-self._k:].mean()-x[:self._k].mean())
        valid = gap < noise_stdev
        return valid

    def _denoise_for_layman(self, sequence, layer):
        '''
        类似于 "expert" 方法，但去噪参数是自动优化的。
        -----
        参数:
            sequence: 1D 浮点数数组
                待去噪的数据序列。
            layer: int
                从相应循环矩阵中选择的前导行数。
        -----
        返回:
            denoised: 1D 浮点数数组
                去噪后的平滑序列。
        -----
        异常:
            AssertionError
                如果条件 1 <= layer <= sequence.size 无法满足。
        '''
        assert 1 <= layer <= sequence.size
        # 代码采用一些邻近数据的平均值来估计序列的边界水平。
        # 默认情况下，这个数字是 11。
        self._k = min(11, sequence.size // 3)
        # 最初，代码假设没有线性倾斜
        trend = np.zeros_like(sequence)
        # 迭代平均长度。
        # 在最坏的情况下，迭代必须在它是 1 时终止
        while self._k >= 1 and not self._cross_validate(sequence - trend, layer):
            self._k -= 2
            trend = np.linspace(0, sequence[-self._k:].mean() - sequence[:self._k].mean(),
                                sequence.size) if self._k >= 1 else 0
        A = self.U[:, :self.r] @ np.diag(self.s[:self.r]) @ self._Vh[:self.r]
        denoised = self._reduce(A) + trend
        return denoised

    def denoise(self, sequence, *args, **kwargs):
        segments = self.split_signal(sequence)
        denoised_segments = [self._method[self.mode](segment, max(1, int(len(segment) / 3)), *args, **kwargs) for segment in segments]
        return self.reconstruct_signal(denoised_segments)


if __name__ == "__main__":
    sequence = np.random.randn(1000)  # Example sequence
    denoiser = Denoiser(mode="layman")
    denoised_sequence = denoiser.denoise(sequence)
    print("Denoising complete.")