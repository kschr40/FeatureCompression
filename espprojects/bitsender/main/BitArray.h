class BitArray {
public:
    BitArray(uint16_t bitCount) : _bitCount(bitCount) {
        _byteCount = (_bitCount + 7) >> 3;

        memset(_data, 0, _byteCount);
    }

    inline bool get(uint16_t index) const {
        return (_data[index >> 3] >> (index & 7)) & 1U;
    }

    inline void set(uint16_t index) {
        _data[index >> 3] |= (1U << (index & 7));
    }

    inline void clear(uint16_t index) {
        _data[index >> 3] &= ~(1U << (index & 7));
    }

    inline void write(uint16_t index, bool value) {
        uint8_t mask = 1U << (index & 7);
        uint8_t& b = _data[index >> 3];
        b = value ? (b | mask) : (b & ~mask);
    }

    inline void toggle(uint16_t index) {
        _data[index >> 3] ^= (1U << (index & 7));
    }

    inline uint16_t size() const {
        return _bitCount;
    }
    
    inline uint8_t* getdata() {
        return _data;
    }

private:
    uint16_t _bitCount;
    uint16_t _byteCount;
    uint8_t  _data[21];  // supports up to 400 bits
};
