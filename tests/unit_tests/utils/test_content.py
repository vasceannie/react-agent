import pytest
import hashlib
import json

class TestCodeUnderTest:

    # Import hashlib and json modules successfully
    def test_modules_import_successfully(self):
        # 1. Import the modules to test if they can be imported
        import hashlib
        import json
    
        # 2. Assert that the modules are of the expected types
        assert isinstance(hashlib, type(json)), "1. Both modules should be of the same type"
        # 3. Check that the modules are not None
        assert hashlib is not None, "2. hashlib module should be imported"
        assert json is not None, "3. json module should be imported"

    # Access hashlib's hash functions like md5, sha1, sha256
    def test_access_hash_functions(self):
        # 1. Import the hashlib module
        import hashlib
    
        # 2. Test accessing different hash functions
        md5_hash = hashlib.md5()  # 1. Create an MD5 hash object
        assert md5_hash is not None, "2. MD5 hash object should be created"
    
        sha1_hash = hashlib.sha1()  # 3. Create a SHA1 hash object
        assert sha1_hash is not None, "4. SHA1 hash object should be created"
    
        sha256_hash = hashlib.sha256()  # 5. Create a SHA256 hash object
        assert sha256_hash is not None, "6. SHA256 hash object should be created"
    
        # 3. Verify that these are different hash algorithms
        assert md5_hash.name != sha1_hash.name, "7. MD5 and SHA1 should have different names"
        assert sha1_hash.name != sha256_hash.name, "8. SHA1 and SHA256 should have different names"

    # Create hash objects and update them with data
    def test_update_hash_objects(self):
        # 1. Import the hashlib module
        import hashlib
    
        # 2. Create a test string
        test_data = "test data to hash"  # 1. Define test data
    
        # 3. Create a hash object
        md5_hash = hashlib.md5()  # 2. Initialize MD5 hash object
    
        # 4. Update the hash object with data
        md5_hash.update(test_data.encode('utf-8'))  # 3. Update hash with encoded data
    
        # 5. Get the hexadecimal digest
        digest1 = md5_hash.hexdigest()  # 4. Get the first digest
        assert isinstance(digest1, str), "5. Digest should be a string"
        assert len(digest1) == 32, "6. MD5 digest should be 32 characters long"
    
        # 6. Update the hash object with more data
        md5_hash.update("more data".encode('utf-8'))  # 7. Update with additional data
    
        # 7. Get the new digest and verify it's different
        digest2 = md5_hash.hexdigest()  # 8. Get the second digest
        assert digest1 != digest2, "9. Digest should change after updating with more data"

    # Import modules in environments where they're not available
    def test_module_import_error_handling(self):
        import sys
        import importlib
    
        # 1. Create a mock module name that doesn't exist
        fake_module_name = "non_existent_module_12345"  # 1. Define a module name that shouldn't exist
    
        # 2. Check if the module is already in sys.modules (unlikely)
        if fake_module_name in sys.modules:  # 2. Check if module exists in sys.modules
            del sys.modules[fake_module_name]  # 3. Remove it if it exists
    
        # 3. Try to import the non-existent module and expect an ImportError
        try:
            # 4. Attempt to import a non-existent module
            importlib.import_module(fake_module_name)
            # 5. If we get here, the test should fail
            assert False, "4. ImportError should be raised for non-existent module"
        except ImportError:
            # 6. This is the expected behavior
            assert True, "5. ImportError was correctly raised"
    
        # 4. Verify that real modules can still be imported
        try:
            # 7. Try importing real modules
            import hashlib
            import json
            assert True, "6. Real modules should import successfully"
        except ImportError:
            # 8. This should not happen
            assert False, "7. Real modules should not raise ImportError"

    # Hash very large data streams efficiently
    def test_hash_large_data_streams(self):
        # 1. Import necessary modules
        import hashlib
        import time
    
        # 2. Create a large data stream (10MB)
        large_data = b"0" * 10_000_000  # 1. Create 10MB of data
    
        # 3. Measure time to hash the data in one go
        start_time = time.time()  # 2. Record start time
    
        # 4. Create hash object and update with all data at once
        single_hash = hashlib.sha256()  # 3. Create SHA256 hash object
        single_hash.update(large_data)  # 4. Update with all data at once
        single_digest = single_hash.hexdigest()  # 5. Get the digest
    
        single_time = time.time() - start_time  # 6. Calculate elapsed time
    
        # 5. Measure time to hash the data in chunks
        start_time = time.time()  # 7. Record start time for chunked approach
    
        # 6. Create hash object and update with chunks
        chunked_hash = hashlib.sha256()  # 8. Create another SHA256 hash object
        chunk_size = 1_000_000  # 9. Define 1MB chunk size
    
        # 7. Process data in chunks
        for i in range(0, len(large_data), chunk_size):  # 10. Iterate through data in chunks
            chunked_hash.update(large_data[i:i+chunk_size])  # 11. Update hash with current chunk
    
        chunked_digest = chunked_hash.hexdigest()  # 12. Get the digest for chunked approach
    
        chunked_time = time.time() - start_time  # 13. Calculate elapsed time for chunked approach
    
        # 8. Verify both approaches produce the same hash
        assert single_digest == chunked_digest, "14. Both hashing approaches should produce the same digest"
    
        # 9. Check that processing was reasonably efficient (no strict timing assertions)
        assert single_time < 1.0, "15. Single update should complete in reasonable time"
        assert chunked_time < 1.0, "16. Chunked updates should complete in reasonable time"

    # Handle non-UTF-8 encodable objects in json serialization
    def test_json_serialization_with_non_utf8_objects(self):
        # 1. Import json module
        import json
    
        # 2. Create a dictionary with various types of data
        test_dict = {
            "string": "normal string",  # 1. Normal string (UTF-8 compatible)
            "number": 42,  # 2. Integer value
            "float": 3.14,  # 3. Float value
            "bool": True,  # 4. Boolean value
            "none": None,  # 5. None value
            "list": [1, 2, 3],  # 6. List value
            "nested": {"key": "value"}  # 7. Nested dictionary
        }
    
        # 3. Test normal serialization
        normal_json = json.dumps(test_dict)  # 8. Serialize normal dictionary
        assert isinstance(normal_json, str), "9. JSON output should be a string"
    
        # 4. Create a problematic object (bytes that aren't UTF-8)
        binary_data = b'\x80\x81\x82'  # 10. Create non-UTF8 binary data
    
        # 5. Add binary data to dictionary
        test_dict["binary"] = binary_data  # 11. Add binary data to dictionary
    
        # 6. Test serialization with default settings (should fail)
        try:
            json.dumps(test_dict)  # 12. Try to serialize with binary data
            assert False, "13. Should raise TypeError for non-serializable binary data"
        except TypeError:
            assert True, "14. Correctly raised TypeError for binary data"
    
        # 7. Test serialization with custom encoder
        def custom_encoder(obj):  # 15. Define custom encoder function
            if isinstance(obj, bytes):  # 16. Check if object is bytes
                return obj.hex()  # 17. Convert bytes to hex string
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")  # 18. Raise error for other types
    
        # 8. Serialize with custom encoder
        custom_json = json.dumps(test_dict, default=custom_encoder)  # 19. Use custom encoder
        assert isinstance(custom_json, str), "20. JSON with custom encoder should be a string"
    
        # 9. Deserialize and check the result
        deserialized = json.loads(custom_json)  # 21. Deserialize the JSON
        assert deserialized["binary"] == binary_data.hex(), "22. Binary data should be converted to hex string"

    # Get hexadecimal digest from hash objects
    def test_get_hexadecimal_digest(self):
        # 1. Create a new hash object using hashlib
        hash_object = hashlib.sha256()
    
        # 2. Update the hash object with a byte string
        hash_object.update(b"OpenAI")
    
        # 3. Get the hexadecimal digest of the hash object
        hex_digest = hash_object.hexdigest()
    
        # 4. Assert that the hexadecimal digest is a string of length 64
        assert isinstance(hex_digest, str), "1. The digest should be a string"
        assert len(hex_digest) == 64, "2. The length of the digest should be 64 characters"

    # Use json.dumps to serialize Python objects to JSON strings
    def test_json_dumps_serialization(self):
        # 1. Import the json module to use its functionality
        import json
    
        # 2. Create a Python dictionary to be serialized
        data = {"name": "Alice", "age": 30}
    
        # 3. Use json.dumps to serialize the dictionary to a JSON string
        json_string = json.dumps(data)
    
        # 4. Assert that the result is a string
        assert isinstance(json_string, str), "1. The result should be a JSON string"
    
        # 5. Assert that the JSON string is correctly formatted
        assert json_string == '{"name": "Alice", "age": 30}', "2. The JSON string should match the expected format"

    # Use json.loads to deserialize JSON strings to Python objects
    def test_json_loads_deserialization(self):
        # 1. Import the json module to use its functionality
        import json
    
        # 2. Define a JSON string to be deserialized
        json_string = '{"name": "John", "age": 30, "city": "New York"}'
    
        # 3. Use json.loads to deserialize the JSON string into a Python dictionary
        result = json.loads(json_string)
    
        # 4. Assert that the result is a dictionary
        assert isinstance(result, dict), "1. The result should be a dictionary"
    
        # 5. Assert that the deserialized data matches the expected dictionary
        expected_result = {"name": "John", "age": 30, "city": "New York"}
        assert result == expected_result, "2. The deserialized data should match the expected dictionary"

    # Parse malformed JSON strings
    def test_parse_malformed_json(self):
        # 1. Define a malformed JSON string
        malformed_json = '{"key": "value", "key2": "value2"'
    
        # 2. Try to parse the malformed JSON string and catch the exception
        try:
            json.loads(malformed_json)
            # 3. If no exception is raised, the test should fail
            assert False, "1. Expected a JSONDecodeError due to malformed JSON"
        except json.JSONDecodeError:
            # 4. If a JSONDecodeError is raised, the test should pass
            pass

    # Handle circular references in Python objects for JSON serialization
    def test_handle_circular_references(self):
        # 1. Create a Python object with a circular reference
        obj = {}
        obj['self'] = obj
    
        # 2. Try to serialize the object using json.dumps
        try:
            json.dumps(obj)
            # 3. If no exception is raised, the test should fail
            assert False, "1. JSON serialization should fail for circular references"
        except (TypeError, ValueError) as e:
            # 4. Check that a TypeError or ValueError is raised
            assert isinstance(e, (TypeError, ValueError)), "2. A TypeError or ValueError should be raised for circular references"

    # Implement data integrity verification with hashlib
    def test_data_integrity_verification(self):
        # 1. Define a sample data dictionary to be serialized and hashed
        sample_data = {'key': 'value'}
    
        # 2. Serialize the data to a JSON formatted string
        serialized_data = json.dumps(sample_data)
    
        # 3. Create a hash of the serialized data using hashlib
        data_hash = hashlib.sha256(serialized_data.encode()).hexdigest()
    
        # 4. Assert that the hash is a string of expected length (64 for SHA-256)
        assert isinstance(data_hash, str), "1. The hash should be a string"
        assert len(data_hash) == 64, "2. The hash length should be 64 characters for SHA-256"
    
        # 5. Verify integrity by re-hashing the same data and comparing hashes
        rehashed_data = hashlib.sha256(serialized_data.encode()).hexdigest()
        assert data_hash == rehashed_data, "3. The hashes should match, indicating data integrity"

    # Serialize complex nested data structures with json
    def test_serialize_complex_nested_data_structures(self):
        # 1. Define a complex nested data structure
        complex_data = {
            "name": "John",
            "age": 30,
            "children": [
                {"name": "Jane", "age": 10},
                {"name": "Doe", "age": 5}
            ],
            "address": {
                "street": "123 Main St",
                "city": "Anytown",
                "zipcode": "12345"
            }
        }
    
        # 2. Serialize the complex data structure using json.dumps
        serialized_data = json.dumps(complex_data)
    
        # 3. Assert that the serialized data is a string
        assert isinstance(serialized_data, str), "1. Serialized data should be a string"
    
        # 4. Deserialize the data back to a Python object
        deserialized_data = json.loads(serialized_data)
    
        # 5. Assert that the deserialized data matches the original data
        assert deserialized_data == complex_data, "2. Deserialized data should match the original data"

    # Parse JSON with custom object hooks using json.JSONDecoder
    def test_json_decoder_with_custom_object_hook(self):
        # 1. Define a custom object hook function that converts JSON objects to Python objects
        def custom_object_hook(dct):
            # 2. If the JSON object has a key 'type' with value 'custom', return a custom object
            if 'type' in dct and dct['type'] == 'custom':
                return CustomObject(dct['value'])
            # 3. Otherwise, return the dictionary as is
            return dct

        # 4. Define a simple class to represent the custom object
        class CustomObject:
            def __init__(self, value):
                self.value = value

        # 5. Create a JSON string that includes an object with a 'type' key
        json_data = '{"type": "custom", "value": "test_value"}'

        # 6. Use json.JSONDecoder with the custom object hook to parse the JSON string
        decoder = json.JSONDecoder(object_hook=custom_object_hook)
        result = decoder.decode(json_data)

        # 7. Assert that the result is an instance of CustomObject
        assert isinstance(result, CustomObject), "1. Result should be an instance of CustomObject"
        # 8. Assert that the value of the custom object is as expected
        assert result.value == "test_value", "2. The value of the custom object should be 'test_value'"

    # Handle custom object serialization with json.JSONEncoder
    def test_custom_object_serialization(self):
        # 1. Define a custom object class
        class CustomObject:
            def __init__(self, name, value):
                self.name = name
                self.value = value
    
        # 2. Create a JSONEncoder subclass to handle CustomObject serialization
        class CustomObjectEncoder(json.JSONEncoder):
            def default(self, obj):
                # 3. Check if the object is an instance of CustomObject
                if isinstance(obj, CustomObject):
                    # 4. Return a dictionary representation of the object
                    return {'name': obj.name, 'value': obj.value}
                # 5. Call the superclass method for other types
                return super().default(obj)
    
        # 6. Create an instance of CustomObject
        custom_obj = CustomObject('test', 123)
    
        # 7. Serialize the custom object using the custom encoder
        serialized = json.dumps(custom_obj, cls=CustomObjectEncoder)
    
        # 8. Assert that the serialized output is as expected
        assert serialized == '{"name": "test", "value": 123}', "1. The serialized output should match the expected JSON format"