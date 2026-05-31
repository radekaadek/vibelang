# Vibelang – Dokumentacja Języka

Vibelang kompilowany język programowania oparty na LLVM.

---

## 1. Typy danych


| Typ | Opis |
| :--- | :--- |
| `int32` | 32-bitowa liczba całkowita ze znakiem. |
| `int64` | 64-bitowa liczba całkowita ze znakiem. |
| `float32` | 32-bitowa liczba zmiennoprzecinkowa (pojedynczej precyzji). |
| `float64` | 64-bitowa liczba zmiennoprzecinkowa (podwójnej precyzji). |
| `bool` | Wartość logiczna (`true` lub `false`). |
| `ID` | Niestandardowy typ użytkownika (nazwa struktury lub klasy). |

---

## 2. Zmienne

### Deklaracja i inicjalizacja

W Vibelang nie istnieje deklaracja zmiennej bez przypisania wartości — każda zmienna musi zostać zainicjalizowana w momencie deklaracji. Deklaracja ma zawsze postać `typ ID = wyrażenie;`.

```vibelang
int32 age = 25;     // Deklaracja i inicjalizacja
float64 pi = 3.14;  // Wartość początkowa jest wymagana
```

### Przypisanie

Aby zmienić wartość istniejącej już zmiennej, używa się samej nazwy, bez podawania typu.

```vibelang
age = 26;
```

---

## 3. Operatory

### Operatory arytmetyczne

Zasady rzutowania: Vibelang automatycznie promuje typy do najwyższej wspólnej precyzji (np. `int32` + `float64` da wynik w `float64`).

* Dodawanie: `+`
* Odejmowanie: `-`
* Mnożenie: `*`
* Dzielenie: `/`

### Operatory relacyjne

Zwracają wartość typu `bool`.

* Mniejsze: `<`
* Mniejsze lub równe: `<=`
* Większe: `>`
* Większe lub równe: `>=`
* Równe: `==`
* Różne: `!=`

### Operatory logiczne

Zwracają wartość typu `bool`. Argumenty są automatycznie konwertowane na typ logiczny.

* Koniunkcja: `and`
* Alternatywa: `or`
* Alternatywa wykluczająca: `xor`
* Negacja: `not`

---

## 4. Wejście / Wyjście (I/O)

Wbudowane funkcje pozwalają na komunikację przez standardowe strumienie.

* **Wypisywanie:** Instrukcja `print(expr);` wypisuje wartość na konsolę.
* **Wczytywanie:** Instrukcja `read(ID);` pobiera wartość z klawiatury i zapisuje ją do wcześniej zadeklarowanej zmiennej.

```vibelang
int32 number = 0;
read(number);
print(number * 2);

```

---

## 5. Instrukcje sterujące

### Instrukcja warunkowa `if`

```vibelang
if x > 10 then
    print(1);
else
    print(0);
end

```

### Pętla `while`

```vibelang
int32 i = 0;
while i < 5 do
    print(i);
    i = i + 1;
end

```

---

## 6. Funkcje

Funkcje definiuje się za pomocą słowa kluczowego `func`. Muszą określać typy parametrów i typ zwracany (lub `void`, jeśli nic nie zwracają).

```vibelang
func add(int32 a, int32 b) -> int32
    return a + b;
end

func greet() -> void
    print(12345); // Przykład akcji
end

```

Wywołania funkcji:

```vibelang
int32 result = add(5, 10);
greet();

```

---

## 7. Struktury (Structs)

Struktury pozwalają grupować dane różnych typów.

### Definicja

```vibelang
struct Point
    float64 x;
    float64 y;
end

```

### Inicjalizacja i dostęp do pól

Struktury inicjalizuje się używając nawiasów klamrowych i dwukropków. Dostęp do pól realizowany jest za pomocą kropki `.`.

```vibelang
Point p = Point { x: 10.0, y: 20.0 };
print(p.x);
p.y = 30.0;

```

---

## 8. Klasy i metody (OOP)

Klasy to rozszerzenie struktur. Oprócz pól (zmiennych członkowskich) mogą zawierać metody.

### Definicja

```vibelang
class Calculator
    int32 memory;

    func init(int32 startVal) -> void
        self.memory = startVal;
    end

    func add(int32 val) -> int32
        self.memory = self.memory + val;
        return self.memory;
    end
end

```

### Tworzenie obiektów i wywoływanie metod

Aby zaalokować obiekt w pamięci, użyj słowa kluczowego `new`. Wewnątrz metod klasy dostępny jest specjalny wskaźnik `self`, który odnosi się do bieżącej instancji.

```vibelang
Calculator calc = new Calculator();
calc.init(0);
int32 result = calc.add(15);
print(result);

```

---

## 9. Komentarze

* **Jednolinijkowe:** Rozpoczynają się od `//` i trwają do końca linii.
* **Wielolinijkowe (blokowe):** Otoczone przez `/*` oraz `*/`.

```vibelang
// To jest komentarz jednolinijkowy

/* To jest komentarz
wielolinijkowy
*/
```
