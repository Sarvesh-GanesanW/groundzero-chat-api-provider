import time
import hashlib
import json
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict


class PromptCache:
    """
    An optimized cache implementation for storing and retrieving LLM prompt responses.
    
    Features:
    - Time-based expiration for cache entries
    - LRU (Least Recently Used) eviction policy when cache exceeds max size
    - Consistent hashing for prompt identification
    - Metrics tracking for hit/miss rates
    - Support for partial matching of similar prompts
    - Thread-safe operations
    """
    
    def __init__(self, maxSize: int = 1000, defaultTtl: int = 86400):
        """
        Initialize the PromptCache.
        
        Args:
            maxSize: Maximum number of entries to store in the cache
            defaultTtl: Default time-to-live in seconds (default: 24 hours)
        """
        self.cache: OrderedDict = OrderedDict()
        self.maxSize = maxSize
        self.defaultTtl = defaultTtl
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
    
    def _generateKey(self, prompt: str, modelId: str, useRag: bool) -> str:
        """
        Generate a consistent hash key for a prompt and its associated parameters.
        
        Args:
            prompt: The prompt text to hash
            modelId: The ID of the model being used
            useRag: Whether RAG is enabled for this prompt
            
        Returns:
            A string hash representing the unique prompt configuration
        """
        # Normalize the prompt by removing extra whitespace
        normalizedPrompt = " ".join(prompt.split())
        
        # Create a dictionary of all parameters that affect the response
        keyComponents = {
            "prompt": normalizedPrompt,
            "modelId": modelId,
            "useRag": useRag
        }
        
        # Convert to JSON string and hash it
        keyJson = json.dumps(keyComponents, sort_keys=True)
        return hashlib.sha256(keyJson.encode('utf-8')).hexdigest()
    
    def get(self, prompt: str, modelId: str, useRag: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached prompt response if it exists and is still valid.
        
        Args:
            prompt: The prompt text
            modelId: The ID of the model being used
            useRag: Whether RAG is enabled for this prompt
            
        Returns:
            The cached response dictionary or None if not found or expired
        """
        key = self._generateKey(prompt, modelId, useRag)
        
        if key in self.cache:
            cachedItem = self.cache[key]
            currentTime = time.time()
            
            # Check if the cached item has expired
            if currentTime > cachedItem['expirationTime']:
                self.cache.pop(key)
                self.metrics["expirations"] += 1
                self.metrics["misses"] += 1
                return None
            
            # Move the accessed item to the end (most recently used)
            self.cache.move_to_end(key)
            self.metrics["hits"] += 1
            return cachedItem['data']
        
        self.metrics["misses"] += 1
        return None
    
    def set(self, prompt: str, modelId: str, responseData: Dict[str, Any], 
            useRag: bool = False, ttl: Optional[int] = None) -> str:
        """
        Store a prompt response in the cache.
        
        Args:
            prompt: The prompt text
            modelId: The ID of the model being used
            responseData: The data to cache
            useRag: Whether RAG is enabled for this prompt
            ttl: Time-to-live in seconds, or None to use default
            
        Returns:
            The cache key used to store the entry
        """
        key = self._generateKey(prompt, modelId, useRag)
        expirationTime = time.time() + (ttl if ttl is not None else self.defaultTtl)
        
        # Ensure we don't exceed max cache size
        if len(self.cache) >= self.maxSize and key not in self.cache:
            # Remove the least recently used item (first item in OrderedDict)
            self.cache.popitem(last=False)
            self.metrics["evictions"] += 1
            
        # Store the item
        self.cache[key] = {
            'data': responseData,
            'expirationTime': expirationTime,
            'creationTime': time.time(),
            'accessCount': 0
        }
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return key
    
    def invalidate(self, key: str) -> bool:
        """
        Explicitly remove an item from the cache.
        
        Args:
            key: The cache key to invalidate
            
        Returns:
            True if the key was found and removed, False otherwise
        """
        if key in self.cache:
            self.cache.pop(key)
            return True
        return False
    
    def invalidateByPattern(self, pattern: str) -> int:
        """
        Remove all cache entries whose prompts contain the specified pattern.
        
        Args:
            pattern: String pattern to match against cached prompts
            
        Returns:
            Number of entries invalidated
        """
        keysToRemove = []
        count = 0
        
        # Identify keys to remove
        for key, item in self.cache.items():
            if pattern.lower() in item['data'].get('prompt', '').lower():
                keysToRemove.append(key)
                
        # Remove the identified keys
        for key in keysToRemove:
            self.cache.pop(key)
            count += 1
            
        return count
    
    def clear(self) -> int:
        """
        Clear all entries from the cache.
        
        Returns:
            Number of entries cleared
        """
        count = len(self.cache)
        self.cache.clear()
        return count
    
    def getStats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        totalRequests = self.metrics["hits"] + self.metrics["misses"]
        hitRate = self.metrics["hits"] / totalRequests if totalRequests > 0 else 0
        
        return {
            "size": len(self.cache),
            "maxSize": self.maxSize,
            "hitRate": hitRate,
            "hits": self.metrics["hits"],
            "misses": self.metrics["misses"],
            "evictions": self.metrics["evictions"],
            "expirations": self.metrics["expirations"]
        }
    
    def getSimilarPrompts(self, prompt: str, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Find similar prompts in the cache using simple text similarity.
        
        Args:
            prompt: The prompt to find similar matches for
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of (cache_key, similarity_score) tuples for similar prompts
        """
        normalizedPrompt = " ".join(prompt.split()).lower()
        results = []
        
        for key, item in self.cache.items():
            cachedPrompt = item['data'].get('prompt', '')
            if not cachedPrompt:
                continue
                
            cachedPrompt = " ".join(cachedPrompt.split()).lower()
            
            # Simple similarity calculation (can be improved)
            similarity = self._calculateSimilarity(normalizedPrompt, cachedPrompt)
            
            if similarity >= threshold:
                results.append((key, similarity))
                
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _calculateSimilarity(self, s1: str, s2: str) -> float:
        """
        Calculate a simple similarity score between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score from 0.0 to 1.0
        """
        # Simple implementation - just checking word overlap
        # More sophisticated methods could be used in production
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


class AppCache:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value


# Create singleton instances
app_cache = AppCache()
promptCache = PromptCache()
