#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
#include <ctype.h>
#include <string.h>

static PyObject *
_base64_encode(const unsigned char *data, Py_ssize_t length)
{
    PyObject *base64_module = PyImport_ImportModule("base64");
    if (!base64_module) {
        return NULL;
    }
    PyObject *b64encode = PyObject_GetAttrString(base64_module, "b64encode");
    Py_DECREF(base64_module);
    if (!b64encode) {
        return NULL;
    }
    PyObject *bytes = PyBytes_FromStringAndSize((const char *)data, length);
    if (!bytes) {
        Py_DECREF(b64encode);
        return NULL;
    }
    PyObject *args = PyTuple_Pack(1, bytes);
    Py_DECREF(bytes);
    if (!args) {
        Py_DECREF(b64encode);
        return NULL;
    }
    PyObject *encoded = PyObject_CallObject(b64encode, args);
    Py_DECREF(args);
    Py_DECREF(b64encode);
    if (!encoded) {
        return NULL;
    }
    PyObject *result = PyUnicode_FromEncodedObject(encoded, "ascii", "strict");
    Py_DECREF(encoded);
    return result;
}

static int
_base64_decode(PyObject *input, unsigned char **out_data, Py_ssize_t *out_len)
{
    PyObject *base64_module = PyImport_ImportModule("base64");
    if (!base64_module) {
        return -1;
    }
    PyObject *b64decode = PyObject_GetAttrString(base64_module, "b64decode");
    Py_DECREF(base64_module);
    if (!b64decode) {
        return -1;
    }
    PyObject *args = PyTuple_Pack(1, input);
    if (!args) {
        Py_DECREF(b64decode);
        return -1;
    }
    PyObject *decoded = PyObject_CallObject(b64decode, args);
    Py_DECREF(args);
    Py_DECREF(b64decode);
    if (!decoded) {
        return -1;
    }
    if (!PyBytes_Check(decoded)) {
        Py_DECREF(decoded);
        PyErr_SetString(PyExc_TypeError, "base64 decode did not return bytes");
        return -1;
    }
    Py_ssize_t len = PyBytes_GET_SIZE(decoded);
    unsigned char *buffer = PyMem_Malloc(len);
    if (!buffer) {
        Py_DECREF(decoded);
        PyErr_NoMemory();
        return -1;
    }
    memcpy(buffer, PyBytes_AS_STRING(decoded), len);
    Py_DECREF(decoded);
    *out_data = buffer;
    *out_len = len;
    return 0;
}

static int
_normalize_fingerprint(PyObject *fingerprint_obj, char **out_data, Py_ssize_t *out_len)
{
    if (!PyUnicode_Check(fingerprint_obj)) {
        PyErr_SetString(PyExc_TypeError, "fingerprint must be a string");
        return -1;
    }
    PyObject *stripped = PyObject_CallMethod(fingerprint_obj, "strip", NULL);
    if (!stripped) {
        return -1;
    }
    const char *raw = PyUnicode_AsUTF8AndSize(stripped, out_len);
    if (!raw) {
        Py_DECREF(stripped);
        return -1;
    }
    if (*out_len == 0) {
        Py_DECREF(stripped);
        PyErr_SetString(PyExc_ValueError, "fingerprint cannot be empty");
        return -1;
    }
    char *buffer = PyMem_Malloc(*out_len);
    if (!buffer) {
        Py_DECREF(stripped);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t i = 0; i < *out_len; i++) {
        buffer[i] = (char)toupper((unsigned char)raw[i]);
    }
    Py_DECREF(stripped);
    *out_data = buffer;
    return 0;
}

static int
_compute_digest_hex(const char *data, Py_ssize_t len, char *output)
{
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) {
        PyErr_SetString(PyExc_RuntimeError, "failed to create digest context");
        return -1;
    }
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) != 1 ||
        EVP_DigestUpdate(ctx, data, (size_t)len) != 1 ||
        EVP_DigestFinal_ex(ctx, digest, &digest_len) != 1) {
        EVP_MD_CTX_free(ctx);
        PyErr_SetString(PyExc_RuntimeError, "failed to compute digest");
        return -1;
    }
    EVP_MD_CTX_free(ctx);
    for (unsigned int i = 0; i < digest_len; i++) {
        sprintf(output + (i * 2), "%02x", digest[i]);
    }
    output[digest_len * 2] = '\0';
    return 0;
}

static PyObject *
_current_hwid_digest(PyObject *self, PyObject *args)
{
    PyObject *fingerprint_obj;
    if (!PyArg_ParseTuple(args, "U", &fingerprint_obj)) {
        return NULL;
    }
    char *normalized;
    Py_ssize_t norm_len;
    if (_normalize_fingerprint(fingerprint_obj, &normalized, &norm_len) < 0) {
        return NULL;
    }
    char hex_digest[EVP_MAX_MD_SIZE * 2 + 1];
    if (_compute_digest_hex(normalized, norm_len, hex_digest) < 0) {
        PyMem_Free(normalized);
        return NULL;
    }
    PyMem_Free(normalized);
    return PyUnicode_FromString(hex_digest);
}

static PyObject *
_derive_encryption_key(PyObject *self, PyObject *args)
{
    PyObject *fingerprint_obj;
    Py_buffer salt_buffer;
    if (!PyArg_ParseTuple(args, "Oy*", &fingerprint_obj, &salt_buffer)) {
        return NULL;
    }
    char *normalized;
    Py_ssize_t norm_len;
    if (_normalize_fingerprint(fingerprint_obj, &normalized, &norm_len) < 0) {
        PyBuffer_Release(&salt_buffer);
        return NULL;
    }
    unsigned char key[EVP_MAX_MD_SIZE];
    unsigned int key_len = 0;
    if (!HMAC(EVP_sha256(), normalized, (int)norm_len,
              (const unsigned char *)salt_buffer.buf,
              salt_buffer.len, key, &key_len)) {
        PyMem_Free(normalized);
        PyBuffer_Release(&salt_buffer);
        PyErr_SetString(PyExc_RuntimeError, "failed to derive key");
        return NULL;
    }
    PyObject *result = PyBytes_FromStringAndSize((const char *)key, (Py_ssize_t)key_len);
    PyMem_Free(normalized);
    PyBuffer_Release(&salt_buffer);
    return result;
}

static PyObject *
_encrypt_license_secret(PyObject *self, PyObject *args, PyObject *kwargs)
{
    const char *secret_buf = NULL;
    Py_ssize_t secret_len = 0;
    PyObject *fingerprint_obj = NULL;
    int file_version = 0;
    static char *kwlist[] = {"secret", "fingerprint", "file_version", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y#U|i", kwlist,
                                     &secret_buf, &secret_len, &fingerprint_obj, &file_version)) {
        return NULL;
    }
    if (secret_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "secret must not be empty");
        return NULL;
    }
    char *normalized;
    Py_ssize_t norm_len;
    if (_normalize_fingerprint(fingerprint_obj, &normalized, &norm_len) < 0) {
        return NULL;
    }
    unsigned char salt[16];
    unsigned char nonce[12];
    if (RAND_bytes(salt, sizeof(salt)) != 1 || RAND_bytes(nonce, sizeof(nonce)) != 1) {
        PyMem_Free(normalized);
        PyErr_SetString(PyExc_RuntimeError, "failed to generate random bytes");
        return NULL;
    }
    unsigned char key[EVP_MAX_MD_SIZE];
    unsigned int key_len = 0;
    if (!HMAC(EVP_sha256(), normalized, (int)norm_len, salt, sizeof(salt), key, &key_len)) {
        PyMem_Free(normalized);
        PyErr_SetString(PyExc_RuntimeError, "failed to derive key");
        return NULL;
    }
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        PyMem_Free(normalized);
        PyErr_SetString(PyExc_RuntimeError, "failed to allocate cipher context");
        return NULL;
    }
    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1 ||
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, sizeof(nonce), NULL) != 1 ||
        EVP_EncryptInit_ex(ctx, NULL, NULL, key, nonce) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        PyMem_Free(normalized);
        PyErr_SetString(PyExc_RuntimeError, "failed to initialize cipher");
        return NULL;
    }
    Py_ssize_t ciphertext_cap = secret_len + 16;
    unsigned char *ciphertext = PyMem_Malloc(ciphertext_cap);
    if (!ciphertext) {
        EVP_CIPHER_CTX_free(ctx);
        PyMem_Free(normalized);
        PyErr_NoMemory();
        return NULL;
    }
    int out_len = 0;
    if (EVP_EncryptUpdate(ctx, ciphertext, &out_len,
                          (const unsigned char *)secret_buf, (int)secret_len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        PyMem_Free(ciphertext);
        PyMem_Free(normalized);
        PyErr_SetString(PyExc_RuntimeError, "encryption failed");
        return NULL;
    }
    int total_len = out_len;
    if (EVP_EncryptFinal_ex(ctx, ciphertext + total_len, &out_len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        PyMem_Free(ciphertext);
        PyMem_Free(normalized);
        PyErr_SetString(PyExc_RuntimeError, "encryption finalize failed");
        return NULL;
    }
    total_len += out_len;
    unsigned char tag[16];
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, sizeof(tag), tag) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        PyMem_Free(ciphertext);
        PyMem_Free(normalized);
        PyErr_SetString(PyExc_RuntimeError, "failed to obtain tag");
        return NULL;
    }
    EVP_CIPHER_CTX_free(ctx);
    unsigned char *buffer = PyMem_Realloc(ciphertext, total_len + sizeof(tag));
    if (!buffer) {
        PyMem_Free(ciphertext);
        PyMem_Free(normalized);
        PyErr_NoMemory();
        return NULL;
    }
    ciphertext = buffer;
    memcpy(ciphertext + total_len, tag, sizeof(tag));
    total_len += sizeof(tag);
    PyObject *salt_b64 = _base64_encode(salt, sizeof(salt));
    PyObject *nonce_b64 = _base64_encode(nonce, sizeof(nonce));
    PyObject *cipher_b64 = _base64_encode(ciphertext, total_len);
    PyMem_Free(ciphertext);
    if (!salt_b64 || !nonce_b64 || !cipher_b64) {
        Py_XDECREF(salt_b64);
        Py_XDECREF(nonce_b64);
        Py_XDECREF(cipher_b64);
        PyMem_Free(normalized);
        return NULL;
    }
    char hex_digest[EVP_MAX_MD_SIZE * 2 + 1];
    if (_compute_digest_hex(normalized, norm_len, hex_digest) < 0) {
        Py_DECREF(salt_b64);
        Py_DECREF(nonce_b64);
        Py_DECREF(cipher_b64);
        PyMem_Free(normalized);
        return NULL;
    }
    PyObject *digest = PyUnicode_FromString(hex_digest);
    PyMem_Free(normalized);
    if (!digest) {
        Py_DECREF(salt_b64);
        Py_DECREF(nonce_b64);
        Py_DECREF(cipher_b64);
        return NULL;
    }
    PyObject *result = PyDict_New();
    if (!result) {
        Py_DECREF(salt_b64);
        Py_DECREF(nonce_b64);
        Py_DECREF(cipher_b64);
        Py_DECREF(digest);
        return NULL;
    }
    PyObject *version_obj = PyLong_FromLong(file_version);
    PyObject *length_obj = PyLong_FromSsize_t(secret_len);
    if (!version_obj || !length_obj ||
        PyDict_SetItemString(result, "version", version_obj) != 0 ||
        PyDict_SetItemString(result, "salt", salt_b64) != 0 ||
        PyDict_SetItemString(result, "nonce", nonce_b64) != 0 ||
        PyDict_SetItemString(result, "ciphertext", cipher_b64) != 0 ||
        PyDict_SetItemString(result, "hwid_digest", digest) != 0 ||
        PyDict_SetItemString(result, "length", length_obj) != 0) {
        Py_XDECREF(version_obj);
        Py_XDECREF(length_obj);
        Py_DECREF(salt_b64);
        Py_DECREF(nonce_b64);
        Py_DECREF(cipher_b64);
        Py_DECREF(digest);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(version_obj);
    Py_DECREF(length_obj);
    Py_DECREF(salt_b64);
    Py_DECREF(nonce_b64);
    Py_DECREF(cipher_b64);
    Py_DECREF(digest);
    return result;
}

static PyObject *
_decrypt_license_secret(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *document = NULL;
    PyObject *fingerprint_obj = NULL;
    int file_version = 0;
    static char *kwlist[] = {"document", "fingerprint", "file_version", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OU|i", kwlist,
                                     &document, &fingerprint_obj, &file_version)) {
        return NULL;
    }
    if (!PyMapping_Check(document)) {
        PyErr_SetString(PyExc_TypeError, "document must be a mapping");
        return NULL;
    }
    PyObject *version_obj = PyMapping_GetItemString(document, "version");
    if (!version_obj) {
        return NULL;
    }
    long version = PyLong_AsLong(version_obj);
    Py_DECREF(version_obj);
    if (version == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (version != file_version) {
        PyErr_SetString(PyExc_ValueError, "Nieobsługiwana wersja zaszyfrowanego sekretu");
        return NULL;
    }
    PyObject *salt_obj = PyMapping_GetItemString(document, "salt");
    PyObject *nonce_obj = PyMapping_GetItemString(document, "nonce");
    PyObject *cipher_obj = PyMapping_GetItemString(document, "ciphertext");
    PyObject *digest_obj = PyMapping_GetItemString(document, "hwid_digest");
    if (!salt_obj || !nonce_obj || !cipher_obj) {
        Py_XDECREF(salt_obj);
        Py_XDECREF(nonce_obj);
        Py_XDECREF(cipher_obj);
        Py_XDECREF(digest_obj);
        PyErr_SetString(PyExc_ValueError, "Dokument sekretu licencji jest uszkodzony");
        return NULL;
    }
    unsigned char *salt_data = NULL;
    Py_ssize_t salt_len = 0;
    unsigned char *nonce_data = NULL;
    Py_ssize_t nonce_len = 0;
    unsigned char *cipher_data = NULL;
    Py_ssize_t cipher_len = 0;
    if (_base64_decode(salt_obj, &salt_data, &salt_len) < 0 ||
        _base64_decode(nonce_obj, &nonce_data, &nonce_len) < 0 ||
        _base64_decode(cipher_obj, &cipher_data, &cipher_len) < 0) {
        Py_DECREF(salt_obj);
        Py_DECREF(nonce_obj);
        Py_DECREF(cipher_obj);
        Py_XDECREF(digest_obj);
        PyMem_Free(salt_data);
        PyMem_Free(nonce_data);
        PyMem_Free(cipher_data);
        return NULL;
    }
    Py_DECREF(salt_obj);
    Py_DECREF(nonce_obj);
    Py_DECREF(cipher_obj);
    if (cipher_len < 16) {
        PyMem_Free(salt_data);
        PyMem_Free(nonce_data);
        PyMem_Free(cipher_data);
        Py_XDECREF(digest_obj);
        PyErr_SetString(PyExc_ValueError, "Ciphertext too short");
        return NULL;
    }
    char *normalized;
    Py_ssize_t norm_len;
    if (_normalize_fingerprint(fingerprint_obj, &normalized, &norm_len) < 0) {
        PyMem_Free(salt_data);
        PyMem_Free(nonce_data);
        PyMem_Free(cipher_data);
        Py_XDECREF(digest_obj);
        return NULL;
    }
    unsigned char key[EVP_MAX_MD_SIZE];
    unsigned int key_len = 0;
    if (!HMAC(EVP_sha256(), normalized, (int)norm_len,
              salt_data, (int)salt_len, key, &key_len)) {
        PyMem_Free(normalized);
        PyMem_Free(salt_data);
        PyMem_Free(nonce_data);
        PyMem_Free(cipher_data);
        Py_XDECREF(digest_obj);
        PyErr_SetString(PyExc_RuntimeError, "failed to derive key");
        return NULL;
    }
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        PyMem_Free(normalized);
        PyMem_Free(salt_data);
        PyMem_Free(nonce_data);
        PyMem_Free(cipher_data);
        Py_XDECREF(digest_obj);
        PyErr_SetString(PyExc_RuntimeError, "failed to allocate cipher context");
        return NULL;
    }
    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1 ||
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, nonce_len, NULL) != 1 ||
        EVP_DecryptInit_ex(ctx, NULL, NULL, key, nonce_data) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        PyMem_Free(normalized);
        PyMem_Free(salt_data);
        PyMem_Free(nonce_data);
        PyMem_Free(cipher_data);
        Py_XDECREF(digest_obj);
        PyErr_SetString(PyExc_RuntimeError, "failed to initialize cipher");
        return NULL;
    }
    Py_ssize_t payload_len = cipher_len - 16;
    unsigned char *plaintext = PyMem_Malloc(payload_len);
    if (!plaintext) {
        EVP_CIPHER_CTX_free(ctx);
        PyMem_Free(normalized);
        PyMem_Free(salt_data);
        PyMem_Free(nonce_data);
        PyMem_Free(cipher_data);
        Py_XDECREF(digest_obj);
        PyErr_NoMemory();
        return NULL;
    }
    int out_len = 0;
    if (EVP_DecryptUpdate(ctx, plaintext, &out_len, cipher_data, (int)payload_len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        PyMem_Free(plaintext);
        PyMem_Free(normalized);
        PyMem_Free(salt_data);
        PyMem_Free(nonce_data);
        PyMem_Free(cipher_data);
        Py_XDECREF(digest_obj);
        PyErr_SetString(PyExc_RuntimeError, "decryption failed");
        return NULL;
    }
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16, cipher_data + payload_len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        PyMem_Free(plaintext);
        PyMem_Free(normalized);
        PyMem_Free(salt_data);
        PyMem_Free(nonce_data);
        PyMem_Free(cipher_data);
        Py_XDECREF(digest_obj);
        PyErr_SetString(PyExc_RuntimeError, "failed to set tag");
        return NULL;
    }
    int total_len = out_len;
    if (EVP_DecryptFinal_ex(ctx, plaintext + total_len, &out_len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        PyMem_Free(plaintext);
        PyMem_Free(normalized);
        PyMem_Free(salt_data);
        PyMem_Free(nonce_data);
        PyMem_Free(cipher_data);
        Py_XDECREF(digest_obj);
        PyErr_SetString(PyExc_ValueError, "Nie udało się odszyfrować sekretu licencji.");
        return NULL;
    }
    total_len += out_len;
    EVP_CIPHER_CTX_free(ctx);
    if (digest_obj && digest_obj != Py_None) {
        char expected_hex[EVP_MAX_MD_SIZE * 2 + 1];
        if (_compute_digest_hex(normalized, norm_len, expected_hex) < 0) {
            PyMem_Free(plaintext);
            PyMem_Free(normalized);
            PyMem_Free(salt_data);
            PyMem_Free(nonce_data);
            PyMem_Free(cipher_data);
            Py_XDECREF(digest_obj);
            return NULL;
        }
        PyObject *expected_obj = PyUnicode_FromString(expected_hex);
        if (!expected_obj) {
            PyMem_Free(plaintext);
            PyMem_Free(normalized);
            PyMem_Free(salt_data);
            PyMem_Free(nonce_data);
            PyMem_Free(cipher_data);
            Py_DECREF(digest_obj);
            return NULL;
        }
        int cmp = PyObject_RichCompareBool(expected_obj, digest_obj, Py_EQ);
        Py_DECREF(expected_obj);
        if (cmp <= 0) {
            PyMem_Free(plaintext);
            PyMem_Free(normalized);
            PyMem_Free(salt_data);
            PyMem_Free(nonce_data);
            PyMem_Free(cipher_data);
            Py_DECREF(digest_obj);
            if (cmp == 0) {
                PyErr_SetString(PyExc_ValueError, "Sekret licencji zapisano dla innego urządzenia");
            }
            return NULL;
        }
    }
    Py_XDECREF(digest_obj);
    PyObject *result = PyBytes_FromStringAndSize((const char *)plaintext, total_len);
    PyMem_Free(plaintext);
    PyMem_Free(normalized);
    PyMem_Free(salt_data);
    PyMem_Free(nonce_data);
    PyMem_Free(cipher_data);
    return result;
}

static PyMethodDef NativeSecurityMethods[] = {
    {"current_hwid_digest", (PyCFunction)_current_hwid_digest, METH_VARARGS, "Calculate normalized hardware digest."},
    {"derive_encryption_key", (PyCFunction)_derive_encryption_key, METH_VARARGS, "Derive encryption key using fingerprint and salt."},
    {"encrypt_license_secret", (PyCFunction)_encrypt_license_secret, METH_VARARGS | METH_KEYWORDS, "Encrypt license secret using AES-GCM."},
    {"decrypt_license_secret", (PyCFunction)_decrypt_license_secret, METH_VARARGS | METH_KEYWORDS, "Decrypt license secret."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef native_security_module = {
    PyModuleDef_HEAD_INIT,
    "_native_security",
    "Native security helpers.",
    -1,
    NativeSecurityMethods
};

PyMODINIT_FUNC
PyInit__native_security(void)
{
    OpenSSL_add_all_algorithms();
    return PyModule_Create(&native_security_module);
}
