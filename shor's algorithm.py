import math
import random
from fractions import gcd
from typing import Optional, Tuple

def mod_exp(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation: (base^exp) mod mod"""
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

def find_period_classical(a: int, N: int) -> Optional[int]:
    """
    Classical period finding (simulates quantum part).
    Finds the smallest r where a^r ≡ 1 (mod N).
    Note: Real quantum algorithm uses Quantum Fourier Transform.
    """
    if gcd(a, N) > 1:
        return None
    
    r = 1
    value = a % N
    
    # Limit search to prevent infinite loop
    max_period = N
    
    while value != 1 and r < max_period:
        value = (value * a) % N
        r += 1
    
    return r if value == 1 else None

def shors_algorithm(N: int, verbose: bool = True) -> Optional[Tuple[int, int]]:
    """
    Shor's algorithm for integer factorization.
    
    Args:
        N: The number to factor
        verbose: If True, print detailed steps
    
    Returns:
        Tuple of (factor1, factor2) if successful, None otherwise
    """
    
    if verbose:
        print(f"Factoring N = {N}")
        print("=" * 50)
    
    # Step 1: Check if N is even
    if N % 2 == 0:
        if verbose:
            print(f"N is even! Factors: 2 and {N // 2}")
        return (2, N // 2)
    
    # Step 2: Check if N is a prime power (N = p^k for k > 1)
    for b in range(2, int(math.sqrt(N)) + 1):
        temp = N
        exp = 0
        while temp % b == 0:
            temp //= b
            exp += 1
        if temp == 1 and exp > 1:
            if verbose:
                print(f"N = {b}^{exp}. Factors: {b} and {N // b}")
            return (b, N // b)
    
    # Step 3: Choose random a in range [2, N-1]
    a = random.randint(2, N - 1)
    if verbose:
        print(f"\nStep 1: Randomly chose a = {a}")
    
    # Step 4: Compute gcd(a, N)
    g = gcd(a, N)
    if verbose:
        print(f"Step 2: gcd({a}, {N}) = {g}")
    
    if g > 1:
        # Lucky! Found a factor immediately
        if verbose:
            print(f"Lucky! Found factor: {g} and {N // g}")
        return (g, N // g)
    
    # Step 5: Find period r (quantum part - simulated classically here)
    if verbose:
        print(f"Step 3: Finding period r where {a}^r ≡ 1 (mod {N})")
        print("        (In real quantum computer: use Quantum Fourier Transform)")
    
    r = find_period_classical(a, N)
    
    if r is None:
        if verbose:
            print("Failed to find period. Try again.")
        return None
    
    if verbose:
        print(f"        Found period: r = {r}")
        # Show the sequence
        seq = [f"{a}^{i} mod {N} = {mod_exp(a, i, N)}" for i in range(min(r + 1, 10))]
        print(f"        Sequence: {', '.join(seq)}")
    
    # Step 6: Check if r is odd
    if r % 2 == 1:
        if verbose:
            print(f"Period r = {r} is odd. Try again with different a.")
        return None
    
    if verbose:
        print(f"Step 4: Period r = {r} is even ✓")
    
    # Step 7: Compute x = a^(r/2) mod N
    x = mod_exp(a, r // 2, N)
    if verbose:
        print(f"Step 5: Computing x = {a}^({r}//2) mod {N} = {x}")
    
    # Check if x ≡ -1 (mod N)
    if x == N - 1:
        if verbose:
            print(f"x ≡ -1 (mod N). Try again with different a.")
        return None
    
    # Step 8: Compute factors using gcd
    factor1 = gcd(x - 1, N)
    factor2 = gcd(x + 1, N)
    
    if verbose:
        print(f"Step 6: Computing factors:")
        print(f"        gcd({x} - 1, {N}) = gcd({x - 1}, {N}) = {factor1}")
        print(f"        gcd({x} + 1, {N}) = gcd({x + 1}, {N}) = {factor2}")
    
    # Return non-trivial factors
    if 1 < factor1 < N:
        if verbose:
            print(f"\n✓ Success! {N} = {factor1} × {N // factor1}")
        return (factor1, N // factor1)
    elif 1 < factor2 < N:
        if verbose:
            print(f"\n✓ Success! {N} = {factor2} × {N // factor2}")
        return (factor2, N // factor2)
    else:
        if verbose:
            print("Failed to find non-trivial factors. Try again.")
        return None

def factor_with_retries(N: int, max_attempts: int = 10, verbose: bool = True) -> Optional[Tuple[int, int]]:
    """
    Run Shor's algorithm with multiple attempts.
    
    Args:
        N: The number to factor
        max_attempts: Maximum number of attempts
        verbose: If True, print detailed steps
    
    Returns:
        Tuple of (factor1, factor2) if successful, None otherwise
    """
    for attempt in range(1, max_attempts + 1):
        if verbose and attempt > 1:
            print(f"\n{'=' * 50}")
            print(f"Attempt {attempt}")
            print('=' * 50)
        
        result = shors_algorithm(N, verbose)
        if result is not None:
            return result
    
    if verbose:
        print(f"\nFailed to factor {N} after {max_attempts} attempts.")
    return None

# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("SHOR'S ALGORITHM - INTEGER FACTORIZATION")
    print("=" * 60)
    print("\nThis algorithm finds prime factors of composite numbers.")
    print("Note: Works best with small odd composite numbers.")
    print("\nSuggested numbers to try: 15, 21, 35, 77, 91, 143, 221")
    print("=" * 60)
    
    while True:
        try:
            # Get input from user
            user_input = input("\nEnter a number to factor (or 'q' to quit): ").strip()
            
            if user_input.lower() == 'q':
                print("\nThank you for using Shor's Algorithm!")
                break
            
            N = int(user_input)
            
            # Validate input
            if N < 3:
                print("Please enter a number greater than 2.")
                continue
            
            if N > 10000:
                print("Warning: Large numbers may take a long time with classical simulation.")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            
            # Ask for verbosity
            verbose_input = input("Show detailed steps? (y/n): ").strip().lower()
            verbose = verbose_input == 'y'
            
            print("\n" + "=" * 60)
            print(f"Factoring N = {N}")
            print("=" * 60)
            
            # Run the algorithm
            result = factor_with_retries(N, max_attempts=10, verbose=verbose)
            
            if result:
                f1, f2 = result
                print(f"\n{'=' * 60}")
                print(f"SUCCESS! {N} = {f1} × {f2}")
                print(f"{'=' * 60}")
                
                # Verify
                if f1 * f2 == N:
                    print("✓ Factorization verified!")
                else:
                    print("✗ Error in factorization!")
            else:
                print(f"\n{'=' * 60}")
                print(f"Failed to factor {N} after 10 attempts.")
                print("Try running again or try a different number.")
                print(f"{'=' * 60}")
        
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
    
    print("\nGoodbye!")
